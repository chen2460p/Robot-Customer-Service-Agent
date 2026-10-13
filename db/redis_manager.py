"""
================================================================================
文件：db/redis_manager.py
职责：Redis 连接管理与缓存操作封装。
      使用连接池，对外提供 get / set / delete 等通用缓存接口，
      以及封装好的 "天气缓存" 和 "RAG 检索结果缓存" 专用方法。
配合：
  - 读取 config/db.yml 中的 redis 配置（由 utils/config_handler.py 的 db_conf 提供）
  - 被 agent/tools/weather_client.py 替换原有的内存字典缓存
  - 被 rag/rag_service.py 新增 RAG 检索结果缓存层
设计要点：
  1. 使用 redis-py 的 ConnectionPool，避免每次操作都新建 TCP 连接。
  2. 模块级单例 redis_manager，import 即可用。
  3. 所有方法都做了异常收敛：Redis 不可用时自动降级，不抛异常给上层 Agent。
     （Redis 是缓存层，挂了不应该阻断主业务流程）
================================================================================
"""

import json
import hashlib

import redis

from utils.config_handler import db_conf
from utils.logger_handler import logger

_redis_conf = db_conf["redis"]


class RedisManager:
    """Redis 连接池管理器：对外暴露通用缓存接口 + 业务专用方法。"""

    def __init__(self):
        # protocol=2：Redis 5.x 只支持 RESP2 协议；Redis 6.0+ 才支持 RESP3（HELLO 命令）
        # 加上 protocol=2 保证新旧 Redis 版本都能连通
        self._pool = redis.ConnectionPool(
            host=_redis_conf["host"],
            port=int(_redis_conf["port"]),
            password=_redis_conf["password"] or None,
            db=int(_redis_conf["db"]),
            max_connections=int(_redis_conf["max_connections"]),
            decode_responses=True,  # 自动把 bytes 解码成 str，避免手动 .decode()
            protocol=2,  # 兼容 Redis 5.x
        )
        self._client = redis.Redis(connection_pool=self._pool)
        self._default_ttl = int(_redis_conf.get("default_ttl", 600))

        # 尝试 ping，Redis 未启动时不阻断 import——缓存挂了不影响主业务
        try:
            self._client.ping()
            logger.info("[Redis] 连接池创建成功，服务可达")
        except Exception as e:
            logger.warning(f"[Redis] 连接失败，缓存功能将不可用: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # 通用缓存接口
    # ------------------------------------------------------------------
    def get(self, key: str) -> str | None:
        """获取缓存值；Redis 不可用时返回 None（不抛异常）。"""
        try:
            return self._client.get(key)
        except Exception as e:
            logger.debug(f"[Redis] get 失败: {key} -> {e}")
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """设置缓存值；ttl 默认用配置文件中的 default_ttl。失败返回 False。"""
        try:
            self._client.set(key, value, ex=ttl or self._default_ttl)
            return True
        except Exception as e:
            logger.debug(f"[Redis] set 失败: {key} -> {e}")
            return False

    def set_json(self, key: str, obj, ttl: int = None) -> bool:
        """把 Python 对象序列化成 JSON 后存入缓存。"""
        try:
            return self.set(key, json.dumps(obj, ensure_ascii=False), ttl)
        except Exception as e:
            logger.debug(f"[Redis] set_json 失败: {key} -> {e}")
            return False

    def get_json(self, key: str):
        """把缓存中的 JSON 字符串反序列化成 Python 对象。"""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"[Redis] get_json 反序列化失败: {key} -> {e}")
            return None

    def delete(self, key: str) -> bool:
        """删除缓存键。"""
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.debug(f"[Redis] delete 失败: {key} -> {e}")
            return False

    def ping(self) -> bool:
        """检测 Redis 是否可达。"""
        try:
            return self._client.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 业务专用：天气缓存
    # ------------------------------------------------------------------
    def weather_cache_key(self, city: str) -> str:
        """生成天气缓存键：weather:geo:深圳、weather:result:深圳"""
        city_hash = hashlib.md5(city.encode()).hexdigest()[:8]
        return f"weather:result:{city_hash}"

    def geo_cache_key(self, city: str) -> str:
        """生成地理编码缓存键"""
        city_hash = hashlib.md5(city.encode()).hexdigest()[:8]
        return f"weather:geo:{city_hash}"

    # ------------------------------------------------------------------
    # 业务专用：RAG 检索结果缓存
    # ------------------------------------------------------------------
    def rag_cache_key(self, query: str) -> str:
        """生成 RAG 缓存键：rag:query:<md5前16位>"""
        q_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        return f"rag:query:{q_hash}"

    def get_rag_cache(self, query: str):
        """获取 RAG 检索缓存（字典格式）。"""
        return self.get_json(self.rag_cache_key(query))

    def set_rag_cache(self, query: str, value: dict, ttl: int = 3600):
        """设置 RAG 检索缓存，默认 1 小时有效。"""
        return self.set_json(self.rag_cache_key(query), value, ttl)


# 模块级单例
redis_manager = RedisManager()
