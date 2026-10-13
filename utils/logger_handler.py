"""
本文件是项目的【统一日志中心】。
提供 get_logger 函数，能同时把日志输出到：
    1) 控制台：默认 INFO 级别，方便运行时实时查看
    2) 文件（按日切割命名）：默认 DEBUG 级别，便于事后排查细节问题

模块级还导出了一个全局单例 logger，业务层只需 `from utils.logger_handler import logger`
即可使用，不用每个文件自己配置 logging。
"""
# logging：Python 标准库日志框架
import logging
# 借用 path_tool 把相对路径 "logs" 转成绝对路径
from utils.path_tool import get_abs_path
# os：用于创建目录
import os
# datetime：日志文件名里用当天日期命名，按日切分
from datetime import datetime

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
# exist_ok=True：如果目录已存在不报错，幂等创建
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  error info debug
# 格式样例：2026-09-21 10:00:00 - agent - INFO - xxx.py:42 - 这里是日志消息
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    """
    创建（或复用）一个名为 name 的 Logger，并挂载控制台 + 文件双通道 Handler。

    设计要点：
        - 控制台输出 INFO 及以上（关键运行信息），文件输出 DEBUG 及以上（含详细调试）
          这种"控制台清爽 / 文件详尽"的双通道设计，既能日常不刷屏，又能事后排查。
        - 通过 `if logger.handlers` 判断避免重复添加 Handler：
          logging.getLogger(name) 对同名 name 返回的是同一个 Logger 单例，
          若每次都 addHandler 会造成同一条日志被重复打印 N 遍。
        - 文件按日期命名（如 agent_20260921.log），天然按日切分，便于按天归档排查。

    :param name: Logger 名称，默认 "agent"
    :param console_level: 控制台最低输出级别，默认 INFO
    :param file_level: 文件最低输出级别，默认 DEBUG
    :param log_file: 自定义日志文件路径，None 时自动按日期生成
    :return: 配置好的 logging.Logger 对象
    """
    # getLogger(name) 是单例：同名返回同一对象，跨模块共享同一份配置
    logger = logging.getLogger(name)
    # 把 Logger 自身的级别设到最低 DEBUG，让所有级别都能流到 Handler，
    # 由各 Handler 自己的 level 决定是否真正输出（更精细的控制）
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    # 如果该 logger 已经挂过 handler（说明之前已经初始化过），直接返回，
    # 防止重复 addHandler 导致同一条日志被打印多次
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler()
    # 控制台只输出 INFO 及以上，DEBUG 这种细碎日志不打到屏幕，避免刷屏
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:        # 日志文件的存放路径
        # 没有显式指定文件路径时，按"名称_YYYYMMDD.log"格式生成当天日志文件
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    # FileHandler 打开文件追加写入；encoding='utf-8' 保证中文日志不乱码
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    # 文件级别设为 DEBUG，把尽可能详细的日志都留到磁盘，便于事后排查问题
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
# 模块级别直接调用一次 get_logger()，拿到全局单例 logger，
# 业务层 import 即用，无需重复配置
logger = get_logger()


if __name__ == '__main__':
    # 自测：直接运行本文件时，演示各级别日志输出
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
