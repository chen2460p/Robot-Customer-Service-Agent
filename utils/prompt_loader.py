"""
本文件是项目的【提示词加载器】，负责把 prompts/ 目录下存放的提示词文本读入内存。

流程：
    1) prompts.yml 里只配置了各类提示词文本文件的【相对路径】
       （如 main_prompt_path、rag_summarize_prompt_path、report_prompt_path）
    2) 本 loader 根据 prompts_conf 拿到这些相对路径，再用 get_abs_path 转成绝对路径
    3) 读取文本文件内容并以字符串形式返回，供 Agent / RAG / 报告生成模块使用

为什么不在 config_handler 启动时就一次性读完提示词？
    - 提示词文本可能很长，且不每个模块都需要全部提示词
    - 按需加载（lazy load）更灵活，便于将来按场景切换不同提示词文件
"""
# 从 config_handler 拿到 prompts.yml 解析后的字典，里面存着各提示词文件的相对路径
from utils.config_handler import prompts_conf
# 借助 path_tool 把相对路径转成绝对路径
from utils.path_tool import get_abs_path
# 复用全局日志器，出错时打印错误信息
from utils.logger_handler import logger


def load_system_prompts():
    """
    加载【主系统提示词】文本（Agent 的角色、人设、行为约束等）。

    :return: 提示词文本字符串
    :raises KeyError: 当 prompts.yml 里没有 main_prompt_path 配置项时
    """
    try:
        # 从配置字典里取提示词文件的相对路径
        # 用 prompts_conf["main_prompt_path"] 而非 .get()，是为了让缺失配置时抛 KeyError
        # 被下面的 except 捕获并打印更友好的错误信息，便于定位是哪个 key 配错了
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        # 这里直接用 open().read()（没 with），属于简化写法：
        # 短文本读完后 Python 解释器会通过 GC 关闭文件，通常不会有句柄泄露问题
        # 但严格来说 with 写法更稳妥，后续可优化
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    """
    加载【RAG 总结/检索提示词】文本（用于让模型基于检索到的资料做总结回答）。

    :return: 提示词文本字符串
    :raises KeyError: 当 prompts.yml 里没有 rag_summarize_prompt_path 配置项时
    """
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    """
    加载【报告生成提示词】文本（用于把对话/检索结果整理成结构化报告输出）。

    :return: 提示词文本字符串
    :raises KeyError: 当 prompts.yml 里没有 report_prompt_path 配置项时
    """
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


if __name__ == '__main__':
    # 自测：直接运行本文件时，打印报告提示词文本
    print(load_report_prompts())
