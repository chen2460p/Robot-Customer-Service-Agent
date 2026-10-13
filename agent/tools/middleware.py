"""
================================================================================================================
文件职责：本文件定义了 ReAct 智能体运行过程中的三个"中间件"（middleware）。
中间件可以理解为"拦截器"——在智能体执行链路上某些关键节点（如调用工具前后、调用模型前、生成提示词时）
自动插入我们写的逻辑，无需改动核心业务代码。

本文件是"动态提示词切换"这一核心亮点的承载者：
    1. 当工具 fill_context_for_report 被执行后，monitor_tool 会向 runtime.context 写入 "report": True 标记。
    2. 下一次模型即将调用、提示词即将生成时，report_prompt_switch 读到这个标记，
       把系统提示词从"客服"切换为"报告写手"，从而让同一个 Agent 在运行中"换身份"。

与谁配合：
    - 与 utils/prompt_loader.py 配合：从那里读取两套不同的系统提示词。
    - 与 langchain.agents.middleware 提供的三个装饰器配合：wrap_tool_call / before_model / dynamic_prompt。
    - 与 fill_context_for_report 工具配合：该工具是"哨兵"，被调用即代表用户希望生成报告。
    - 与 ReAct Agent 主体配合：这些中间件会在 Agent 主流程的固定时机被自动调用。
================================================================================================================
"""

# typing.Callable：用于类型注解，表示 handler 是一个"可调用对象"（函数）
from typing import Callable
# load_system_prompts：加载默认的"客服系统提示词"；load_report_prompts：加载"报告写手系统提示词"
from utils.prompt_loader import load_system_prompts, load_report_prompts
# AgentState：智能体运行时维护的整体状态结构，最核心的是 messages 列表
from langchain.agents import AgentState
# 三个核心装饰器：
#   wrap_tool_call   —— 包装工具调用，可在工具执行前后插入逻辑
#   before_model     —— 在每次调用大模型之前执行
#   dynamic_prompt   —— 在每次生成系统提示词之前执行，可用返回值替换默认提示词
# ModelRequest：dynamic_prompt 中间件接收到的请求对象，封装了 runtime、state 等
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
# ToolCallRequest：wrap_tool_call 中间件收到的请求封装，里面包含工具名、参数、runtime 等
from langchain.tools.tool_node import ToolCallRequest
# ToolMessage：工具执行后产生的消息类型；Command：某些场景下工具需要返回跳转指令而不是普通消息
from langchain_core.messages import ToolMessage
# Runtime：LangGraph 运行时上下文，里面的 context 字典可以跨中间件、跨节点传递自定义标记
from langgraph.runtime import Runtime
# Command：用于让工具直接控制图跳转的指令对象（这里用作返回类型注解之一）
from langgraph.types import Command
# logger：项目统一封装的日志器，方便统一日志格式
from utils.logger_handler import logger


# @wrap_tool_call：装饰器把本函数注册为"工具调用中间件"，每个工具被 Agent 调用前后都会进入这里。
# 设计意图：统一记录工具调用日志 + 在特定工具（哨兵工具）执行后向 context 写标记。
@wrap_tool_call
def monitor_tool(
        # 请求的数据封装：包含本次工具调用的名字、参数、runtime 上下文等
        request: ToolCallRequest,
        # 执行的函数本身：调用它即等于"真正去执行工具"，类似"放行到下一层"
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    # 工具执行前：先打印正在执行的工具名，便于排查"调到了哪个工具"
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    # 工具执行前：再打印入参，便于排查"参数是否被模型正确填写"
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        # 调用 handler 表示"放行"——真正去执行工具本体，拿到结果
        result = handler(request)
        # 工具执行成功后：记录一条成功日志
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # ===== 核心亮点：哨兵工具识别 + 写入上下文标记 =====
        # 如果刚执行的工具是 fill_context_for_report，说明用户意图是"生成报告"，
        # 此时向 runtime.context 写入一个布尔标记，后续 report_prompt_switch 会读取它来切换提示词。
        # 这样设计的好处：不需要修改 Agent 主体代码，仅靠中间件就完成了"工具执行 → 状态标记 → 提示词切换"的全链路联动。
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        # 把工具结果正常返回给主流程
        return result
    except Exception as e:
        # 任何工具异常都打一条 error 日志，便于定位失败工具
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        # 异常继续向上抛，让上层看到工具确实失败了（不吞掉异常）
        raise e


# @before_model：装饰器把本函数注册为"模型调用前中间件"，每次 Agent 要调用大模型之前都会进来一次。
# 设计意图：在调模型前打印日志，方便观察"对话此时有多少消息、最后一条是什么"。
# 返回 None 表示不修改任何东西、不阻断流程，只是观察。
@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    # 打印即将调用模型时对话里有多少条消息，能直观看到对话进展
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    # 打印最后一条消息的类型和内容，便于排查"模型这次会基于什么输入做推理"
    # 这里用 debug 级别，避免正常运行时输出太多正文
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    # 返回 None：明确表示不修改状态、不拦截，让模型正常被调用
    return None


# @dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
# 设计意图：根据 runtime.context 中的 "report" 标记，动态决定本次对话用哪一套系统提示词。
# 这就是"动态切换提示词"的实现点：客服 ↔ 报告写手。
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    # 从运行时上下文读取 report 标记；如果还没被设置过（默认客服场景），用 False 兜底
    is_report = request.runtime.context.get("report", False)
    # 如果标记为 True：说明前一步的 fill_context_for_report 已经把上下文置为"报告模式"，
    # 这里返回报告写手专用的系统提示词，模型下一次的"人设"就变成报告写手了。
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    # 默认场景：返回客服系统提示词，维持原有的客服身份
    return load_system_prompts()
