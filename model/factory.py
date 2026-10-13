"""
================================================================================================================
文件职责：本文件用"抽象工厂模式"统一管理项目里两类模型的创建——对话模型（chat_model）和嵌入模型（embed_model）。
提供模块级单例对象供整个项目复用，避免到处重复实例化、重复加载配置。

为什么用工厂模式：
    - 把"具体用什么模型、叫什么名字、走哪个 SDK"这些易变细节集中在一处，调用方只面对统一的 generator() 接口；
    - 以后想换模型（如换成 qwen3.8-max、换成开源模型）只需要改本文件一行，业务代码零改动；
    - 通过抽象基类 BaseModelFactory 强制子类实现 generator()，结构清晰、不易遗漏。

为什么用模块级单例：
    - 模型对象创建开销大（要初始化远程连接、加载配置），项目里只需要一份；
    - 模块级变量 chat_model / embed_model 在首次 import 本模块时就被创建一次，全局共享同一份对象，
      后续 RAG / Agent 直接 import 即可拿到，免去重复 new 的成本。

【2026-10-13 架构变更】
    ChatModelFactory 从 ChatTongyi（旧版 DashScope API，仅支持 qwen-max/plus 等经典模型）
    切换为 ChatOpenAI（走 DashScope 兼容模式 /compatible-mode/v1，支持 qwen3.8-max 等新模型）。
    嵌入模型仍用 DashScopeEmbeddings，因为 text-embedding-v4 在旧版 API 上可用。

与谁配合：
    - utils/config_handler.py：通过 rag_conf 读取模型名称等配置。
    - rag/vector_store.py：VectorStoreService 建库时需要 embed_model（默认导入本模块的 embed_model）。
    - rag/rag_service.py：RagSummarizeService 直接 import 本模块的 chat_model 用于 LCEL 链的总结环节。
    - agent 主体：通过 chat_model 与大模型交互。
================================================================================================================
"""

# abc.ABC / abc.abstractmethod：Python 标准库抽象基类工具。
from abc import ABC, abstractmethod
# os：读取 DASHSCOPE_API_KEY 环境变量（ChatOpenAI 需要显式传入 api_key）
import os
# Optional：类型注解，表示返回值可以为 None
from typing import Optional
# Embeddings：LangChain 中嵌入模型的统一抽象基类，给类型注解用
from langchain_core.embeddings import Embeddings
# BaseChatModel：LangChain 中对话模型的统一抽象基类
from langchain_core.language_models.chat_models import BaseChatModel
# DashScopeEmbeddings：阿里 DashScope 提供的嵌入模型 LangChain 封装
from langchain_community.embeddings import DashScopeEmbeddings
# ChatOpenAI：LangChain 的 OpenAI 兼容接口实现——DashScope 兼容模式完全兼容 OpenAI API 协议
# 通过 base_url 指向 DashScope 兼容模式端点，即可调用 qwen3.8-max 等仅兼容模式支持的新模型
from langchain_openai import ChatOpenAI
# rag_conf：项目统一的 RAG 配置读取器，里面含 chat_model_name、embedding_model_name 等键
from utils.config_handler import rag_conf


# 抽象工厂基类：定义统一的"造模型"接口，所有具体工厂都要继承它并实现 generator()
class BaseModelFactory(ABC):
    # @abstractmethod：强制子类必须实现该方法，否则子类无法被实例化
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 基类只声明接口，不写实现，交给子类决定到底造出哪一种模型
        pass


# DashScope 兼容模式端点——qwen3.x 系列新模型只在这个 API 上可用
_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 具体工厂之一：专门负责造"对话模型"
# 【变更】从 ChatTongyi（旧版 DashScope API）切换到 ChatOpenAI（兼容模式）
# 原因：qwen3.8-max 等新模型只在 DashScope 兼容模式 /compatible-mode/v1 上可用，
# ChatTongyi 底层走 dashscope.Generation SDK 旧版 API，会报 400 url error
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=_DASHSCOPE_COMPAT_BASE,
        )


# 具体工厂之二：专门负责造"嵌入模型"
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# ===== 模块级单例：在本模块被第一次 import 时执行一次，全局共享 =====
# chat_model：对话模型单例，给 Agent 和 RAG 总结链用
chat_model = ChatModelFactory().generator()
# embed_model：嵌入模型单例，主要给 VectorStoreService 建库/检索时用
embed_model = EmbeddingsFactory().generator()
