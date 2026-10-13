"""
================================================================================
文件：agent/react_agent.py
职责：把"模型 + 系统提示词 + 工具集 + 中间件"组装成一个可流式调用的 ReAct 智能体。
配合：
  - 上游：被 app.py 调用（execute_stream 是给前端的入口方法）。
  - 下游：调用 model/factory.py 拿通义千问模型、调用 utils/prompt_loader.py 加载系统提示词、
          调用 agent/tools/agent_tools.py 里的 7 个 @tool 工具、
          调用 agent/tools/middleware.py 里的 3 个中间件。
设计要点：
  1. ReAct = Reason + Act：模型先"思考"再决定调用哪个工具，工具返回结果后继续思考，
     直到得出最终答案。这一切由 langchain 的 create_agent 编排，开发者只需提供组件。
  2. context={"report": False} 是一个跨中间件共享的运行时标记：
     - 初始 False 表示当前是"普通客服模式"。
     - 当模型调用 fill_context_for_report 这个"哨兵工具"后，
       中间件 report_prompt_switch 会把它改成 True，
       从而在下一次调模型前把系统提示词从"客服"切到"报告写手"。
================================================================================
"""

# create_agent 是 LangChain 官方提供的"组装函数"，把模型/提示词/工具/中间件拼成一个 Agent
from langchain.agents import create_agent
# AIMessage：模型自己产出的消息类型。用于把"模型说的话"与"用户消息/工具结果"区分开，
# 避免把用户提问和工具原始输出也流式吐到前端（见 execute_stream 中的说明）
from langchain_core.messages import AIMessage
# chat_model 是本项目封装好的通义千问 qwen3-max 模型实例（来自 model/factory.py）
from model.factory import chat_model
# load_system_prompts 从配置文件加载系统提示词（可能包含客服/报告两套提示词）
from utils.prompt_loader import load_system_prompts
# agent_conf：读取多轮对话的上下文窗口配置
from utils.config_handler import agent_conf
# 导入 7 个 @tool 装饰过的工具函数：RAG检索、天气、用户位置、用户ID、当前月份、外部数据、哨兵工具
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
# 导入 3 个中间件：监控工具调用、调模型前打日志、根据 context 切换提示词
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    """
    ReAct 智能体的对外封装类。

    为什么封装成一个类而不是一堆裸函数？
    ——因为 Agent 实例（self.agent）在创建时要加载模型、注册工具、绑定中间件，
      代价较高且可复用。封装成类后，前端 app.py 只需 new 一次 ReactAgent，
      后续多轮对话都复用同一个实例，性能更好。
    """

    def __init__(self):
        """
        构造时立即组装好内部的 self.agent。
        组装四要素：
          - model：底层 LLM，决定"大脑"是谁（这里用通义千问）。
          - system_prompt：系统提示词，决定"角色身份 / 行为准则"。
          - tools：智能体可调用的工具集合，模型会自主决定何时调、调哪个。
          - middleware：中间件链，在模型调用前后注入额外逻辑（日志、监控、提示词切换）。
        """
        self.agent = create_agent(
            # 底层大模型实例——通义千问 qwen3-max
            model=chat_model,
            # 系统提示词——告诉模型"你是扫地机器人客服，应该怎么回答"
            system_prompt=load_system_prompts(),
            # 工具列表——模型会根据用户问题自主选择调用哪些工具
            # 7 个工具依次：RAG检索、天气、用户位置、用户ID、当前月份、外部数据、哨兵工具
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            # 中间件——在 Agent 运行过程中的关键节点（工具调用前后、模型调用前后）插入自定义逻辑
            # 顺序一般按"先监控→后切换"的依赖关系排
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

        # 多轮对话上下文窗口：最多携带最近 N 条历史消息
        # 为什么需要窗口？—— 历史会随对话无限增长，持续推高 token 消耗最终撑爆上下文。
        # 截断到最近 N 条是成本与效果之间最简单可控的折中；
        # 更复杂的方案（历史摘要压缩、按相关性检索历史）留待后续按需引入。
        self.max_history_messages = int(agent_conf.get("max_history_messages", 10))

    def execute_stream(self, query: str, history: list[dict] | None = None):
        """
        对外暴露的流式调用入口。

        参数：
            query: str —— 用户这一轮的提问文本。
            history: list[dict] | None —— 之前的多轮对话历史，形如
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]。
                     传 None 即单轮调用（无上下文）。

        返回：
            一个生成器（generator），每次 yield 出一段【模型自己产出】的文本。
            前端（app.py）用 write_stream 消费这个生成器实现流式打字机效果。

        设计要点：
          - 输入用 {"messages": [...]} 的字典结构，因为 LangChain 的 Agent
            内部以"消息列表"为统一上下文格式（兼容多轮、工具消息等）。
          - stream_mode="values" 表示每次 yield 整个状态快照（包含 messages 列表），
            方便我们取最后一条消息；与之对应的还有 "updates"（只吐变更增量）等模式。
          - context={"report": False} 是跨中间件共享的运行时标记，详见文件头说明。
        """
        # 组装消息列表 = 历史（截断到窗口大小）+ 本轮的 user 消息
        messages: list[dict] = []

        if history:
            # 只保留 role/content 都合法的条目，避免前端传入脏数据导致 Agent 报错
            valid = [m for m in history
                     if m.get("role") in ("user", "assistant") and m.get("content")]
            # 取【最近】N 条——越近的对话对当前这轮影响越大，且能控制 token 消耗
            messages.extend(valid[-self.max_history_messages:])

        messages.append({"role": "user", "content": query})
        input_dict = {"messages": messages}

        # 用于去重：stream_mode="values" 可能把同一状态快照重复吐出，
        # 记录上一条已输出的消息 id，避免同一段文字被流式输出两次
        last_emitted_id = None

        # stream() 是 Agent 的流式调用入口：
        #   - input_dict：本轮输入（历史 + 当前提问）
        #   - stream_mode="values"：每次吐完整状态快照（不是增量），便于直接取 messages[-1]
        #   - context={"report": False}：运行时上下文，初始为"非报告模式"
        #     中间件 report_prompt_switch 会读取/修改这个字典，实现"客服↔报告写手"提示词切换
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            # 每个 chunk 是一次状态快照，messages 是最新消息列表
            latest_message = chunk["messages"][-1]

            # 【只输出模型自己说的话】——必须过滤掉另外两类消息，否则前端会看到脏内容：
            #   1) 第一条快照的 messages[-1] 是【用户刚发的提问】。不过滤的话，助手气泡会
            #      先把用户的问题原样重复一遍（因为它是"回答的第一个分片"）；
            #   2) 工具执行后快照的 messages[-1] 是 ToolMessage。不过滤的话，检索到的原始
            #      资料、原始天气串会直接倒进聊天界面。
            # 两者都只出现在中间状态，正式的客服回答只来自 AIMessage。
            if not isinstance(latest_message, AIMessage):
                continue
            if not latest_message.content:
                continue

            # 同一快照被重复吐出时 id 相同，跳过以避免重复输出
            if latest_message.id:
                if latest_message.id == last_emitted_id:
                    continue
                last_emitted_id = latest_message.id

            # .strip() 去掉首尾空白，+ "\n" 让前端换行更整齐
            yield latest_message.content.strip() + "\n"


# 这一段是"直接运行本文件时"的简易测试入口，方便开发时不用启 Streamlit 就能调试 Agent
if __name__ == '__main__':
    agent = ReactAgent()

    # 用"给我生成我的使用报告"这种触发哨兵工具的提问来验证提示词切换是否生效
    print("=" * 30, "① 报告场景（验证提示词切换）", "=" * 30)
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        # end="" 让 print 不换行，flush=True 强制立即刷新输出（流式打印必备）
        print(chunk, end="", flush=True)

    # 多轮对话验证：第二轮的追问依赖第一轮的历史，若无历史传入模型必然答不出
    print("\n", "=" * 30, "② 多轮对话（验证上下文传递）", "=" * 30)
    history = [
        {"role": "user", "content": "我叫蒲小明，请记住我的名字。"},
        {"role": "assistant", "content": "好的，蒲小明！我已经记住你的名字了。"},
    ]
    for chunk in agent.execute_stream("我叫什么名字？", history=history):
        print(chunk, end="", flush=True)
