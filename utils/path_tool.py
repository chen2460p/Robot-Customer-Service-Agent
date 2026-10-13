"""
为整个工程提供统一的绝对路径。
本文件是项目的【路径地基】，被 config_handler / logger_handler / file_handler 等几乎所有模块依赖。

为什么需要"以自身位置推算根目录"？
    - 用户可能从项目根、也可能从某个子目录启动脚本（如 python agent/main.py）
    - 如果用相对路径（如 "config/rag.yml"）会被解读为相对于【当前工作目录】，
      切换启动目录就找不到文件。
    - 解决办法：以【本文件自身的绝对路径】为锚点向上推算项目根，
      因为不管从哪启动，path_tool.py 自身的物理位置是固定的，所以根目录也是固定的。
"""
# os：提供路径处理能力（abspath / dirname / join 等）
import os


def get_project_root() -> str:
    """
    获取工程所在的根目录。

    实现思路：以【本文件 path_tool.py 自身】的物理路径为基准，向上跳一级目录。
        假设项目结构为：
            项目根/
            ├── utils/
            │   └── path_tool.py   ← __file__ 指向这里
            ├── config/
            ├── prompts/
            └── ...
        那么 os.path.dirname 一次得到 utils/ 目录，再 dirname 一次就得到项目根。

    :return: 字符串根目录（绝对路径）
    """
    # 当前文件的绝对路径
    # __file__ 是 Python 内置变量，指向"当前正在执行的脚本文件路径"，
    # 用 abspath 确保转换为绝对路径（避免某些情况下 __file__ 是相对路径）
    current_file = os.path.abspath(__file__)
    # 获取工程的根目录，先获取文件所在的文件夹绝对路径
    # dirname 取父目录：从 .../utils/path_tool.py 得到 .../utils
    current_dir = os.path.dirname(current_file)
    # 获取工程根目录
    # 再 dirname 一次：从 .../utils 得到 ...（即项目根目录）
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径，得到绝对路径。

    用法示例：
        get_abs_path("config/rag.yml")     -> 项目根/config/rag.yml 的绝对路径
        get_abs_path("prompts/system.txt") -> 项目根/prompts/system.txt 的绝对路径

    :param relative_path: 相对领（相对项目根目录的路径，使用 / 分隔）
    :return: 绝对路径
    """
    # 先拿到项目根目录，再拼接业务传入的相对路径，得到跨启动目录可用的绝对路径
    project_root = get_project_root()
    # os.path.join 会自动处理路径分隔符（Windows 用 \，Linux 用 /），
    # 所以传入 "config/rag.yml" 也能在两个平台正确拼出路径
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    # 自测：直接运行本文件时，打印一个示例配置文件的绝对路径
    print(get_abs_path("config/config.txt"))
