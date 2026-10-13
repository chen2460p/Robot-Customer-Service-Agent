"""
================================================================================
文件：agent/tools/agent_tools.py
职责：定义智能体可调用的 7 个 @tool 工具函数 + 一个惰性加载外部数据的辅助函数。
配合：
  - 被 agent/react_agent.py 在 create_agent(tools=...) 时注册，模型会自主决定调用哪个。
  - 被 agent/tools/middleware.py 中的中间件在运行时拦截/观察。
  - 内部调用 rag/rag_service.py（真实 RAG 检索）、utils/config_handler.py（读配置）、
    utils/path_tool.py（拼绝对路径）、utils/logger_handler.py（打日志）。
设计要点：
  1. 每个工具用 @tool 装饰器声明，装饰器参数 description 是给模型看的"工具说明"——
     模型根据这段描述判断"该不该调这个工具"。
  2. 工具分为两类：
     - 【真实实现】rag_summarize（真实向量检索）、get_weather（真实调用 Open-Meteo 公开 API）、
       fetch_external_data（真实读取 records.csv）；
     - 【模拟实现】get_user_location / get_user_id / get_current_month 仍返回 mock 数据
       （真实业务中用户身份应从登录态或 Token 获取，本项目简化处理）。
  3. fetch_external_data 使用"惰性加载"模式：第一次调用时才把 records.csv 读进内存，
     后续调用直接复用字典缓存，避免每次都重读文件。
  4. fill_context_for_report 是"哨兵工具"——它本身不做业务，
     但被调用后中间件会修改运行时 context["report"]=True，从而触发提示词切换。
================================================================================
"""

import os  # os.path.exists 用来检查外部数据文件是否存在
from utils.logger_handler import logger  # 项目统一的日志器，比 print 更可控（带级别、可落盘）
from langchain_core.tools import tool  # @tool 装饰器：把普通函数变成 LangChain 识别的工具，自动抽取签名和描述
from rag.rag_service import RagSummarizeService  # 真正的 RAG 检索服务（向量库 Chroma + 通义千问 Embedding）
import random  # random.choice 用来从 mock 数据列表里随机抽一个，模拟"返回数据"
from utils.config_handler import agent_conf  # 项目配置读取器（读取 settings 里的 external_data_path 等键）
from utils.path_tool import get_abs_path  # 把配置里的相对路径转成绝对路径，方便 open() 使用
from agent.tools.weather_client import WeatherClient  # 真实天气客户端（Open-Meteo，免密钥）

# MySQL + Redis 连接管理器（阶段 1/2 引入的新依赖）
# 如果 MySQL/Redis 服务未启动，import 时会尝试连接；
# 但我们在 fetch_external_data 中做了降级兜底，不会阻断主流程
try:
    from db.mysql_manager import mysql_manager
    _MYSQL_AVAILABLE = True
except Exception as e:
    logger.warning(f"[agent_tools] MySQL 不可用，将降级为 CSV 读取: {type(e).__name__}")
    _MYSQL_AVAILABLE = False

try:
    from db.redis_manager import redis_manager
    _REDIS_AVAILABLE = True
except Exception as e:
    logger.warning(f"[agent_tools] Redis 不可用，将跳过缓存层: {type(e).__name__}")
    _REDIS_AVAILABLE = False

rag = RagSummarizeService()
# RAG 检索服务实例：在模块加载时就构造好（内部连 Chroma 向量库）
# 后续 rag_summarize 工具直接调用它的方法做"检索 + 摘要"

weather_client = WeatherClient()
# 天气客户端实例：模块级单例，内部复用 requests.Session（保持连接、省掉重复 TLS 握手）
# 并持有两级 TTL 缓存，因此必须全局共用一份，否则缓存无法生效

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
# mock 的用户ID池——get_user_id 工具会从这里随机挑一个返回

month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]
# mock 的月份池——get_current_month 工具从这里随机挑一个返回

external_data = {}
# 外部数据的"内存缓存字典"，初始为空
# 第一次调用 fetch_external_data → generate_external_data() 时才被填充
# 这就是"惰性加载（lazy load）"：用到时才加载，避免启动时就吃内存


@tool(description="从向量存储中检索参考资料")
# description 是给模型看的——模型根据这句话判断"用户的提问是否需要从知识库检索"
def rag_summarize(query: str) -> str:
    """
    真正的 RAG 工具：把用户问题交给 RAG 服务做"向量检索 + 摘要"。

    参数：
        query: str —— 需要在知识库里检索的问题/关键词。
    返回：
        str —— 检索到的参考资料摘要文本，会作为工具结果回传给模型。
    设计：
        这里只是"转发"，真正逻辑在 RagSummarizeService 内部
        （Embedding → Chroma 相似度检索 → 拼接 → 返回）。
    """
    return rag.rag_summarize(query)


@tool(description="获取指定城市的实时天气，返回温度、湿度、风速、空气质量与降雨概率")
def get_weather(city: str) -> str:
    """
    真实天气查询：调用 Open-Meteo 公开 API（免密钥）获取指定城市的实时天气。

    参数：
        city: str —— 城市名称（中文或拼音，由地理编码接口识别）。
    返回：
        str —— 中文天气描述；真实 API 不可用时返回配置中的降级文案。
    设计：
        HTTP 请求、超时重试、两级缓存都在 WeatherClient 内部完成，
        本工具只负责"发起调用 + 决定降级策略"，保持工具层职责单一。
        工具层刻意不抛异常——天气是非核心工具，不应因它挂掉而中断整个 Agent 流程。
    """
    result = weather_client.get_weather(city)
    if result is not None:
        return result

    # 降级：网络超时 / 重试用尽 / 城市名无法识别时，返回明确的不可用提示，
    # 让模型如实告诉用户"我查不到"，而不是编造一份看似合理的假天气。
    logger.warning(f"[get_weather] 城市{city}天气获取失败，返回降级文案")
    return weather_client.fallback_text


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    随机返回一个用户所在城市（mock）。
    无入参——因为"当前用户的城市"在本教学项目中不需要真查询定位。
    返回：城市名字符串。
    """
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """
    随机返回一个用户ID（mock）。
    无入参——真实业务里通常从登录态/Token 里取，这里简化成随机挑。
    返回：user_ids 列表中的某一个。
    """
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """
    随机返回一个月份字符串（mock，格式 "YYYY-MM"）。
    无入参。
    返回：month_arr 中的某一个，例如 "2025-03"。
    """
    return random.choice(month_arr)


# ---------------------------------------------------------------------------
# 【降级兜底】当 MySQL/Redis 都不可用时，回退到原有的 CSV 惰性加载逻辑
# 这一段是"安全网"，保证数据库挂了 Agent 也能正常回答使用报告问题
# ---------------------------------------------------------------------------
def _generate_external_data_from_csv():
    """从 records.csv 解析 external_data 字典（惰性加载，仅 MySQL 不可用时走这条路）。"""
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过表头
                arr = [x.replace('"', "") for x in line.strip().split(",")]
                if len(arr) != 6:
                    continue
                user_id, feature, efficiency, consumables, comparison, time_ = arr
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time_] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    查询某个用户某个月的使用记录（特征/效率/耗材/对比）。

    参数：
        user_id: str —— 用户ID
        month: str —— 月份字符串，如 "2025-03"
    返回：
        找到的记录（dict 类型，由 LangChain 序列化成字符串给模型）；
        找不到则返回空字符串 ""。

    数据源优先级：Redis 缓存 → MySQL 查询 → CSV 文件降级
    """
    # --- 第 1 层：Redis 缓存 ---
    if _REDIS_AVAILABLE:
        cache_key = f"usage:{user_id}:{month}"
        cached = redis_manager.get_json(cache_key)
        if cached is not None:
            logger.info(f"[fetch_external_data] 命中 Redis 缓存: {user_id}/{month}")
            return cached

    # --- 第 2 层：MySQL 查询 ---
    if _MYSQL_AVAILABLE:
        row = mysql_manager.query_one(
            "SELECT feature, efficiency, consumables, comparison "
            "FROM user_usage_records WHERE user_id = %s AND month = %s",
            (user_id, month),
        )
        if row is not None:
            result = {
                "特征": row["feature"],
                "效率": row["efficiency"],
                "耗材": row["consumables"],
                "对比": row["comparison"],
            }
            # 写入 Redis 缓存（1 小时有效）
            if _REDIS_AVAILABLE:
                redis_manager.set_json(f"usage:{user_id}:{month}", result, ttl=3600)
            logger.info(f"[fetch_external_data] 命中 MySQL: {user_id}/{month}")
            return result
        logger.info(f"[fetch_external_data] MySQL 无此记录，尝试 CSV 降级: {user_id}/{month}")

    # --- 第 3 层：CSV 降级 ---
    _generate_external_data_from_csv()
    try:
        result = external_data[user_id][month]
        logger.info(f"[fetch_external_data] 命中 CSV 降级数据: {user_id}/{month}")
        return result
    except KeyError:
        logger.warning(f"[fetch_external_data] 所有数据源均未找到: {user_id}/{month}")
        return ""


@tool(description="无入参，调用后返回固定确认字符串；触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """
    "哨兵工具"——本身不做任何业务，只用于"被调用"这个动作。

    为什么需要哨兵工具？
      - 我们想实现：当模型决定"要写报告"时，自动把系统提示词从"客服"切到"报告写手"。
      - 但模型本身不能直接改 context，只能通过"调用工具"这个动作发出信号。
      - 所以设置一个空壳工具 fill_context_for_report：模型一调它，
        中间件 report_prompt_switch 就能检测到这次调用，进而修改 context["report"]=True，
        下一次调模型前会据此切换系统提示词。
      - 这是一种"用工具调用作为状态切换信号"的常用 Agent 设计模式。
    无入参，返回固定字符串（让模型知道哨兵已触发）。
    """
    return "fill_context_for_report已调用"
