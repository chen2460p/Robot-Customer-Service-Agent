"""
================================================================================
文件：app.py
职责：本项目（"扫地机器人智能客服"）的前端入口，基于 Streamlit 实现聊天界面。
配合：与 agent/react_agent.py 配合——本文件只负责"界面 + 流式展示"，
      真正的 Agent 推理、工具调用、提示词切换都在 ReactAgent 内部完成。
设计要点：
  1. Streamlit 每次用户交互都会从头重跑整个脚本，所以必须用
     st.session_state 缓存 Agent 实例和对话历史，避免重复构造、避免历史丢失。
  2. 使用流式输出 + 打字机效果（逐字 yield），让用户看到模型正在"思考"，
     提升交互体验，符合聊天类应用惯例。
================================================================================
"""

import time  # time.sleep 用于打字机效果——每输出一个字符停顿 0.01 秒，模拟真人打字

import streamlit as st  # Streamlit 是 Python 的轻量级 Web UI 框架，写完 .py 直接 streamlit run 即可成网页
from agent.react_agent import ReactAgent  # 导入本项目自己实现的 ReAct 智能体封装类，下方用它构造 Agent 实例

# 页面顶部大标题 + 分割线，纯 UI 装饰
# 标题
st.title("智扫通机器人智能客服")
st.divider()

# st.session_state 是 Streamlit 提供的"会话级字典"
# 只要用户不刷新浏览器/不关页面，里面的内容在每次脚本重跑时都会保留
# 为什么必须用 session_state 缓存 Agent？
#   - Agent 实例内部加载了通义千问模型、注册了 7 个工具、绑定了 3 个中间件，构造代价很高。
#   - 如果每次重跑都 new 一个新的，会非常慢甚至重复初始化向量库连接。
#   - 放进 session_state 后，整个会话期间只构造一次，后续多轮对话复用同一个实例。
# 第一次进入页面时，session_state 里还没有 "agent"，于是构造一个 ReactAgent 实例存进去
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 同样地，"message" 用来保存整个对话历史（user/assistant 交替）
# 为什么必须缓存历史？
#   - Streamlit 重跑脚本时，普通变量会被重置，对话历史就丢了。
#   - 存进 session_state 后，刷新/重跑仍能回放之前的聊天内容。
# 初始化为空列表，后续每轮对话都会 append 新消息
if "message" not in st.session_state:
    st.session_state["message"] = []

# 页面刷新/重跑后，先把历史对话"回放"出来，否则用户看不到之前的聊天记录
for message in st.session_state["message"]:
    # chat_message 会渲染成聊气泡（user 右侧、assistant 左侧）
    # .write() 把内容塞进气泡
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
# st.chat_input 在页面底部渲染一个输入框，用户回车后返回输入文本；没输入则返回 None
prompt = st.chat_input()

# 只有用户真的输入了内容，才进入"提问→回答"的流程
if prompt:
    # 先把用户这句话立刻渲染到右侧的 user 气泡里，让用户有即时反馈
    st.chat_message("user").write(prompt)
    # 同时把这条 user 消息追加到对话历史，下一轮重跑时上面那个 for 循环能回放它
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 这个列表用来收集模型流式返回的"分片"（chunk），最后要拼成完整回答存进历史
    response_messages = []
    # 在 Agent 思考期间，页面正中央显示一个 spinner 动画 + 提示文字
    with st.spinner("智能客服思考中..."):
        # 调用 Agent 的流式接口，拿到的是一个生成器（generator），逐段吐出文本
        # 【多轮对话】把之前的对话历史一并传进去，否则 Agent 每轮都是无状态的，
        # 用户问"那它呢""刚才说的那个"这类指代性追问会直接失效。
        # 这里用 [:-1] 排除掉刚追加的本轮 user 消息——execute_stream 内部会自己补上当前提问，
        # 不排除的话同一句话会在消息列表里出现两次。
        history = st.session_state["message"][:-1]
        res_stream = st.session_state["agent"].execute_stream(prompt, history=history)

        # 定义一个内部函数 capture：
        # 它一边消费上面的生成器（拿到一段 chunk），一边把 chunk 暂存到 cache_list，
        # 同时把 chunk 拆成单个字符，每吐一个字符就 sleep 0.01 秒，制造打字机效果。
        # 为什么这么做？——流式输出本身已经分片，但单次 chunk 可能还是一大段；
        # 逐字 + sleep 让前端感受到"逐字浮现"的细腻动效，更像真人客服回复。
        def capture(generator, cache_list):

            for chunk in generator:  # 遍历 Agent 返回的每一个分片
                cache_list.append(chunk)  # 把分片存进缓存列表，供后面拼接完整答案使用

                for char in chunk:  # 在当前 chunk 内部，再按字符逐个吐出
                    time.sleep(0.01)  # 每个字符之间停顿 0.01 秒，制造打字机效果
                    yield char  # yield 让这个函数本身也变成生成器，逐字供给 write_stream

    # write_stream 接受一个生成器，边生成边把内容追加到 assistant 气泡里
    # 用 capture 包一层后，assistant 气泡里就会出现"打字机"动效
    st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
    # response_messages[-1] 是最后一段流式输出，作为本轮完整答案存进历史
    # 加个空值兜底：万一本轮没有任何模型文本产出（例如只发生工具调用），
    # 直接取 [-1] 会抛 IndexError 把页面搞崩，这里退化成空串更稳。
    st.session_state["message"].append(
        {"role": "assistant", "content": response_messages[-1] if response_messages else ""})
    # st.rerun() 强制让 Streamlit 立即从头重跑一次脚本
    # 为什么需要 rerun？——因为本次是在 if prompt 分支里渲染气泡，
    # 重跑后，上面的"回放历史"循环就能把这条 assistant 消息按正常顺序重新渲染，
    # 保证页面布局和滚动行为正确。
    st.rerun()
