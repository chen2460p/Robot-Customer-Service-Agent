# 向量库服务：负责把本地知识文档加载、分片、向量化并写入 Chroma 向量库，
# 同时对外提供检索器（retriever）供 RAG 检索总结服务（RagSummarizeService）使用。

# Chroma 向量库的 LangChain 封装，提供持久化存储与相似度检索能力
from langchain_chroma import Chroma
# LangChain 文档统一抽象：page_content 存文本，metadata 存来源等元数据
from langchain_core.documents import Document
# 读取 chroma.yml 配置（集合名、持久化目录、分片参数、数据路径、MD5 存储路径等）
from utils.config_handler import chroma_conf
# 向量化模型单例（text-embedding-v4），文档入库时用它把文本转成向量
from model.factory import embed_model
# 递归字符分片器：按分隔符层级递归切分，尽量保持语义完整
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 把相对路径转成基于项目根目录的绝对路径，保证任意目录启动都能找到文件
from utils.path_tool import get_abs_path
# 文件处理工具：PDF/txt 加载器、按后缀过滤目录、计算文件 MD5（用于去重）
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
# 统一日志器：控制台 INFO + 按日文件 DEBUG
from utils.logger_handler import logger
import os


class VectorStoreService:
    """向量库服务：封装 Chroma 的初始化、知识库构建（load_document）与检索（get_retriever）。"""

    def __init__(self):
        # 初始化 Chroma 向量库实例
        # collection_name: 集合名（类似数据库表名），不同集合相互隔离
        # embedding_function: 向量化模型，写入/查询时都会用它把文本转成向量
        # persist_directory: 持久化目录，向量数据落盘位置，重启后不丢失
        #
        # 【修复①】persist_directory 必须经 get_abs_path 锚定到项目根后再交给 Chroma。
        # 原实现直接把配置里的相对路径 "chroma_db" 传给 Chroma，Chroma 会按【当前工作目录 CWD】
        # 解析它，于是从不同目录启动就落进不同的库：
        #     python app.py（项目根）→ ./chroma_db       （空库）
        #     进入 rag/ 启动        → ./rag/chroma_db   （有数据的库）
        # 后果非常隐蔽：应用照常启动、客服照常回答，但检索恒为空、RAG 静默失效且不报错。
        # 统一锚定项目根后，无论从哪个目录启动都指向同一个库。
        self.persist_dir = get_abs_path(chroma_conf["persist_directory"])
        # 确保持久化目录存在——下面 MD5 记录文件要写进这个目录，不能依赖 Chroma 的隐式创建
        os.makedirs(self.persist_dir, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=self.persist_dir,
        )

        # 递归字符分片器：把长文档切成小块，每块作为一个独立向量入库
        # chunk_size: 每块最大字符数（200）
        # chunk_overlap: 相邻块重叠字符数（20），保证跨块语义不被生硬截断
        # separators: 分隔符层级，按顺序优先在高级分隔符（段落>换行>句号）处切分，
        #             兼顾中英文标点，使切分更贴合自然语言边界
        # length_function: 用 len 计算长度（按字符数而非 token 数）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        # 把向量库转为检索器；k=3 表示每次返回相似度最高的 3 个文档片段
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _md5_store_path(self) -> str:
        """
        MD5 去重记录文件的绝对路径——与向量库放在【同一个持久化目录】下。

        【修复②】修复①暴露出的连带问题：去重记录与向量库是两个各自独立的生命周期。
        原实现把记录文件固定放在项目根（md5.text），而向量库在 persist_directory。
        只要 persist_directory 发生变化（换目录、清库重建、CWD 不同），
        记录文件仍声称"这些文件已入库"，于是新库对所有文件一律跳过 ——
        结果是库是空的、却没有任何一条报错，问题极难定位。
        把记录文件收进持久化目录后，二者同生共死：换库 = 记录自动归零 = 自动重新入库。
        """
        return os.path.join(self.persist_dir, chroma_conf["md5_hex_store"])

    def load_document(self):
        """
        知识库构建主流程（离线执行一次即可）：
        扫描数据文件夹 → 按文件类型加载 → MD5 去重 → 分片 → 向量化入库 → 记录 MD5。
        通过 MD5 记录已入库文件，实现增量更新：新文档放入 data/ 后再跑一次即可。
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            """检查某文件的 MD5 是否已记录过（即该文件是否已入库），True=已处理可跳过。"""
            md5_store_path = self._md5_store_path()

            # MD5 记录文件不存在时，创建空文件并返回“未处理过”
            if not os.path.exists(md5_store_path):
                # 创建文件（立即 close 以释放句柄，仅用于占位）
                open(md5_store_path, "w", encoding="utf-8").close()
                return False            # md5 没处理过

            # 逐行读取已记录的 MD5，命中即说明该文件已入库
            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过（已入库）

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            """把已成功入库文件的 MD5 追加写入记录文件，供下次去重判断。"""
            with open(self._md5_store_path(), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """根据文件后缀选择对应的 LangChain 加载器，返回 Document 列表；不支持的类型返回空列表。"""
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # 扫描数据目录，只保留允许的知识库文件类型（如 txt、pdf）
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 遍历每个待入库文件
        for path in allowed_files_path:
            # 获取文件的 MD5，作为该文件内容的唯一指纹
            md5_hex = get_file_md5_hex(path)

            # 已入库则跳过，避免重复向量化（节省时间与向量库空间）
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                # 加载文件为 Document 列表
                documents: list[Document] = get_file_documents(path)

                # 文件内无有效文本时跳过（如空文件或解析失败）
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 分片：长文档切成多个小块 Document
                split_document: list[Document] = self.spliter.split_documents(documents)

                # 分片后仍无内容时跳过
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将分片向量化并写入 Chroma（embedding_function 在内部自动调用）
                self.vector_store.add_documents(split_document)

                # 入库成功后记录 MD5，下次运行即跳过该文件
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info=True 会记录完整堆栈，便于排查加载失败原因；
                # continue 保证单个文件失败不会中断整体入库流程
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


# 直接运行本文件时的自测入口：先构建知识库，再用检索器验证检索效果
if __name__ == '__main__':
    vs = VectorStoreService()

    # 第一步：把 data/ 下的文档加载入库（已入库的会自动跳过）
    vs.load_document()

    # 第二步：获取检索器并做一次检索验证
    retriever = vs.get_retriever()

    # 用示例 query 检索相似文档片段
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
