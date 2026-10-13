
"""
================================================================================================================
文件职责：本文件实现 RAG 检索 + 总结服务 RagSummarizeService。
用户提问后，先用向量库检索 top-3 相关资料，再把"问题 + 参考资料"一起交给大模型总结出回复。

业务流程（典型 RAG 思路）：
    用户提问 → 向量检索找相关文档 → 把文档拼进提示词 → 调用大模型总结回答

技术亮点：
    1. 使用 LangChain LCEL（LangChain Expression Language）的管道风格组装链：
       PromptTemplate → print_prompt → chat_model → StrOutputParser
       竖线 | 表示"上一步的输出作为下一步的输入"，可读性高、易扩展。
    2. print_prompt 是我们插入到链中间的一个"调试钩子"——把要发给模型的最终提示词打印出来，
       方便开发期排查"模型到底看到了什么"。

与谁配合：
    - rag/vector_store.py 的 VectorStoreService：负责建/加载 Chroma 向量库，本服务拿到 retriever 用于检索。
    - utils/prompt_loader.py：通过 load_rag_prompts() 读取 RAG 专用的提示词模板。
    - model/factory.py：导入模块级单例 chat_model 作为总结环节的大模型。
================================================================================================================
"""
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
# Document：LangChain 中"一段文本 + 元数据"的统一封装类型；检索结果就是 Document 列表
from langchain_core.documents import Document
# StrOutputParser：LangChain 的输出解析器，把模型返回的 AIMessage 自动转成纯字符串，方便直接 print 或返回前端
from langchain_core.output_parsers import StrOutputParser
# VectorStoreService：负责 Chroma 向量库的创建/加载，本服务调用它的 get_retriever() 拿到检索器
from rag.vector_store import VectorStoreService
# load_rag_prompts：从外部加载 RAG 总结专用的提示词模板文本（含 {input} 和 {context} 两个占位符）
from utils.prompt_loader import load_rag_prompts
# PromptTemplate：LangChain 的提示词模板，支持 {变量名} 占位符，被 invoke 时按字典填充
from langchain_core.prompts import PromptTemplate
# chat_model：模块级单例对话模型，从工厂模式产出，避免重复实例化
from model.factory import chat_model
# logger：统一日志器，用于 Redis 缓存命中/写入时记录
from utils.logger_handler import logger
# Redis 缓存管理器（阶段 2 引入，为 RAG 检索结果加缓存层）
try:
    from db.redis_manager import redis_manager
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False


# 一个"调试用"的小工具：把它插进 LCEL 链的中间，用于把传给模型的最终提示词打印出来观察。
# 关键点：必须把传入的 prompt 原样 return 出去，否则会把链后面的环节断掉（这里返回的是带 to_string() 的 PromptValue，符合链下游的输入预期）。
def print_prompt(prompt):
    # 用等号分隔线，便于在终端里一眼看出这一段就是发给模型的最终提示词
    print("="*20)
    # to_string()：把 PromptValue 渲染成纯文本，方便阅读
    print(prompt.to_string())
    print("="*20)
    # 原样返回，让链能继续往下走（→ chat_model → StrOutputParser）
    return prompt


# RAG 总结服务类：把"检索 + 提示词 + 模型 + 解析"组合到一起，对外暴露 rag_summarize 入口
class RagSummarizeService(object):
    # 构造方法：在创建实例时就完成所有"重资源"的初始化——向量库、retriever、提示词模板、LCEL 链
    # 这样设计的好处：rag_summarize 被频繁调用时不必每次重新建链，只跑链即可，性能更好
    def __init__(self):
        # 创建向量库服务实例（负责加载/连接 Chroma）
        self.vector_store = VectorStoreService()
        # 通过向量库拿到检索器（retriever），后续 invoke(query) 即可拿到相关文档
        self.retriever = self.vector_store.get_retriever()
        # 加载 RAG 专用提示词模板文本（含 {input} 和 {context} 两个变量占位符）
        self.prompt_text = load_rag_prompts()
        # 用模板文本构造 PromptTemplate 对象，后续可被链 invoke 时填充变量
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 复用全局单例 chat_model，避免每个服务实例都自己 new 一个模型
        self.model = chat_model
        # 装配 LCEL 链，链只要装配一次就可以多次 invoke
        self.chain = self._init_chain()

    # 私有方法：用 LCEL 管道语法装配出"检索后总结"的链
    # 链顺序：prompt_template → print_prompt → model → StrOutputParser
    #   1) prompt_template：把 {input}/{context} 变量填充进模板，生成最终提示词
    #   2) print_prompt：调试钩子，把上面生成的最终提示词打印出来便于排错（不改变内容）
    #   3) model：把提示词喂给大模型，得到 AIMessage
    #   4) StrOutputParser：把 AIMessage 解析成纯字符串
    # 整条链就是"输入字典 → 提示词 → 模型 → 字符串"的流水线
    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    # 给 query，返回检索到的相关文档列表（list[Document]）
    # 检索器由 VectorStoreService 内部配置（默认 top_k=3），无需在这里再指定数量
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    # RAG 总结的主入口：用户提问 query，返回模型总结后的字符串回复
    def rag_summarize(self, query: str) -> str:
        # ===== 第 0 步：Redis 缓存层（阶段 2 引入）=====
        # 相同 query 的回答直接命中缓存，省去向量检索 + LLM 调用开销
        if _REDIS_AVAILABLE:
            cached = redis_manager.get_rag_cache(query)
            if cached is not None:
                logger.info(f"[RAG] 命中 Redis 缓存: {query}")
                return cached

        # 第一步：用问题去向量库检索相关参考资料
        context_docs = self.retriever_docs(query)

        # 第二步：把检索到的多个 Document 拼成一段长文本，作为模型回答时的"参考资料"
        # 给每段资料加编号（【参考资料1】【参考资料2】...），便于模型在回答时引用、也便于人查看检索质量
        # 同时把 metadata 也带上，让模型能看到来源信息（比如所属文档/页码），从而生成更可追溯的回答
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 拼接格式：编号 + 正文内容 + 元数据；\n 用于每段资料换行，便于阅读
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 第三步：把"用户问题"和"参考资料"一起作为变量传给链，链会自动填充到 PromptTemplate，
        # 经 print_prompt 打印、交给 chat_model 总结、最后被 StrOutputParser 解析成字符串返回。
        # 这里 input 对应模板里的 {input}，context 对应模板里的 {context}
        result = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

        # ===== 第 4 步：写入 Redis 缓存 =====
        if _REDIS_AVAILABLE:
            redis_manager.set_rag_cache(query, result, ttl=3600)
            logger.info(f"[RAG] 结果已写入 Redis 缓存: {query}")

        return result


# 本文件直接被 python 运行时的小测试：用一个示例问题验证整条 RAG 链路是否能正常总结
if __name__ == '__main__':
    # 创建 RAG 总结服务实例（内部会建链、连向量库）
    rag = RagSummarizeService()

    # 用一个具体问题跑一遍完整流程，打印模型总结的回复
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
