"""
本文件是项目【文件处理工具集】，给 RAG 知识库构建阶段使用：
    - get_file_md5_hex         : 算文件 MD5 指纹，用于判断文件是否重复（避免同一文件重复入库）
    - listdir_with_allowed_type: 列出指定目录下符合后缀要求的文件（如只取 .pdf .txt）
    - pdf_loader / txt_loader  : 把 PDF / TXT 文件解析为 LangChain Document 对象，供后续切块向量化

主要被 rag 模块（构建知识库 / 入库 / 检索）调用。
"""
# os：标准库，提供路径判断、目录遍历等系统能力
import os
# hashlib：标准库，提供 MD5 / SHA 等哈希算法，本文件用其算文件指纹去重
import hashlib
# 复用本项目日志器，统一打印 info / error 日志，便于排查问题
from utils.logger_handler import logger
# LangChain 文档对象基类：所有 loader 加载出的内容都统一封装成 Document，
# 包含 page_content（正文文本）和 metadata（来源/页码等元信息）
from langchain_core.documents import Document
# LangChain 社区提供的现成文档加载器：
#   PyPDFLoader 专门解析 PDF；TextLoader 专门读取纯文本文件。
# 选用现成 loader 而非自己写解析逻辑，是因为它们已经处理好分页、编码等细节
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str):     # 获取文件的md5的十六进制字符串
    """
    计算文件的 MD5 十六进制指纹字符串。

    设计要点：
        - 用 4KB 分块读取，而不是一次性 read() 整个文件，
          这样几 GB 的大文件也不会撑爆内存。
        - 用 "rb" 二进制模式读取，因为 MD5 需要原始字节而非文本
          （文本模式会因换行/编码转换导致指纹不稳定）。
        - 返回的 md5_hex 字符串可作为文件"身份证号"，
          相同内容 → 相同 MD5 → 判定重复。

    :param filepath: 文件路径
    :return: 32 位十六进制 MD5 字符串；文件不存在/出错时返回 None
    """
    # 前置校验：文件是否存在，避免后面 open 时报错难以排查
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    # 再次校验：路径必须是文件而非目录，避免对目录做读取
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    # 创建一个 MD5 哈希对象，后续通过 update 喂入数据
    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片，避免文件过大爆内存
    try:
        # with 保证文件句柄最终一定被关闭，即使在读取过程中发生异常
        # "rb"：二进制读取模式，是哈希计算的正确做法
        with open(filepath, "rb") as f:     # 必须二进制读取
            # 海象运算符 := ：先执行 f.read(chunk_size) 赋值给 chunk，
            # 同时把读到的内容作为 while 条件；读到文件末尾时 chunk 为空字节串 → 循环退出
            while chunk := f.read(chunk_size):
                # 把当前 4KB 块喂给 MD5，MD5 是增量算法，
                # 多次 update 等价于一次性算整个文件
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            # 全部喂完后调 hexdigest() 得到最终的 32 位十六进制字符串
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        # 任何异常都兜底记录并返回 None，避免调用方直接崩溃
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):        # 返回文件夹内的文件列表（允许的文件后缀）
    """
    列出指定目录下后缀符合要求的文件绝对路径。

    用途：RAG 构建知识库时只关心 .pdf/.txt 等可解析文件，
          用此函数一次性筛掉无关文件（如 .md .docx 等暂不支持的格式）。

    :param path: 目录路径
    :param allowed_types: 允许的后缀元组，如 (".pdf", ".txt")
    :return: 符合条件文件的绝对路径元组；目录无效时返回 allowed_types（保留原逻辑）
    """
    files = []

    # 校验目录有效性
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    # os.listdir 只返回文件名（不含目录路径），所以下面用 os.path.join 拼回完整路径
    for f in os.listdir(path):
        # str.endswith 接收元组时，只要命中其中任一后缀即返回 True，
        # 比写一堆 if 高效简洁
        if f.endswith(allowed_types):
            # 拼成绝对路径返回，方便后续 loader 直接读取
            files.append(os.path.join(path, f))

    # 返回元组而非列表：元组不可变，更安全，且可作字典 key / 集合元素
    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """
    使用 LangChain PyPDFLoader 加载 PDF 文件为 Document 列表。

    :param filepath: PDF 文件路径
    :param passwd: 加密 PDF 的密码，无密码传 None
    :return: Document 列表，每个 Document 对应一页
    """
    # PyPDFLoader 默认按页切分，每页 → 一个 Document，metadata 里带页码
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    """
    使用 LangChain TextLoader 加载纯文本文件为 Document 列表。

    :param filepath: txt 文件路径
    :return: 包含一个 Document 的列表（整篇文本作为一个 Document）
    """
    # 显式指定 utf-8 编码，避免 Windows 默认 GBK 解码中文 txt 报 UnicodeDecodeError
    return TextLoader(filepath, encoding="utf-8").load()
