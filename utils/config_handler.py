"""
本文件是整个"扫地机器人智能客服"项目的【配置加载中心】。
负责把项目里 4 份 YAML 配置文件（config/rag.yml、chroma.yml、prompts.yml、agent.yml）
一次性读入内存，解析成 Python 字典，并导出 4 个全局配置对象：
    - rag_conf     : RAG 检索增强生成相关配置（如对话模型名、检索参数）
    - chroma_conf  : Chroma 向量数据库相关配置（如向量库路径、embedding 模型）
    - prompts_conf: 提示词文本文件路径配置
    - agent_conf  : Agent 智能体相关配置（如工具、模型、流程）

业务层（agent/rag/model 等模块）只需 `from utils.config_handler import rag_conf`
即可拿到配置，无需自己再去读文件。配置在 import 时即加载，保证全局统一。
"""
# yaml 是 Python 标准库之外的第三方库，用于解析 YAML 配置文件为 Python 字典
import yaml
# 从本项目 path_tool 借用 get_abs_path：把"config/xxx.yml"这种相对路径变成绝对路径，
# 这样无论从哪个目录启动脚本，都能正确找到配置文件
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    """
    加载 RAG（检索增强生成）相关配置。

    :param config_path: 配置文件路径，默认指向项目根目录下的 config/rag.yml
    :param encoding: 文件编码，默认 utf-8（防止中文乱码）
    :return: 解析后的 Python 字典（如 {"chat_model_name": "gpt-4", ...}）
    """
    # 使用 with 上下文管理器打开文件：文件读取完毕或发生异常时，会自动关闭文件句柄，
    # 避免资源泄露（对比直接 open() 需要手动 close()）
    with open(config_path, "r", encoding=encoding) as f:
        # yaml.load 把 YAML 文本解析成 Python 对象（字典/列表）
        # Loader=yaml.FullLoader 安全加载器，能解析 YAML 标准类型但不执行任意 Python 代码，
        # 比默认的 Loader 更安全（避免 yaml.load 不指定 Loader 引发的安全警告）
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    """
    加载 Chroma 向量数据库相关配置（向量库存储路径、embedding 模型等）。

    :param config_path: 默认指向 config/chroma.yml
    :param encoding: 文件编码，默认 utf-8
    :return: 解析后的 Python 字典
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    """
    加载提示词路径配置。
    prompts.yml 里只存放各类提示词文本文件（system_prompt/rag_prompt/report_prompt）的相对路径，
    真正的提示词文本由 prompt_loader.py 在需要时再读取。

    :param config_path: 默认指向 config/prompts.yml
    :param encoding: 文件编码，默认 utf-8
    :return: 解析后的 Python 字典
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    """
    加载 Agent 智能体相关配置（智能体使用的模型、可用工具列表、思考流程等）。

    :param config_path: 默认指向 config/agent.yml
    :param encoding: 文件编码，默认 utf-8
    :return: 解析后的 Python 字典
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_db_config(config_path: str=get_abs_path("config/db.yml"), encoding: str="utf-8"):
    """
    加载数据库与缓存连接配置（MySQL 连接参数、Redis 连接参数等）。

    :param config_path: 默认指向 config/db.yml
    :param encoding: 文件编码，默认 utf-8
    :return: 解析后的 Python 字典
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ============================================================================
# 模块级全局配置对象：在【import 本模块时立即执行】，全项目只需读一次配置文件。
# 这样设计的好处：
#   1) 业务代码 import 时即可拿到现成的配置，无需每次手动调用 load_xxx_config()
#   2) 避免运行期间反复读磁盘，提高性能
#   3) 配置全项目统一，避免散落各处读取导致不一致
# 副作用：如果 YAML 文件缺失或语法错误，import 阶段就会抛异常（早暴露早修复）
# ============================================================================
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
db_conf = load_db_config()


if __name__ == '__main__':
    # 直接运行本文件时做最小自检：打印 rag.yml 里的 chat_model_name 和 db.yml 的配置
    print(rag_conf["chat_model_name"])
    print(db_conf)
