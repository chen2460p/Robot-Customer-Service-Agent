"""
================================================================================
文件：db/mysql_manager.py
职责：MySQL 数据库连接管理与 CRUD 操作封装。
      使用连接池管理连接，提供通用的查询、插入、更新接口。
配合：
  - 读取 config/db.yml 中的 mysql 配置（由 utils/config_handler.py 的 db_conf 提供）
  - 被 db/init_db.py 调用进行建表与数据迁移
  - 被 agent/tools/agent_tools.py 调用替代原有的 CSV 读取逻辑
设计要点：
  1. 使用 pymysql + DBUtils 连接池，避免每次操作都新建 TCP 连接。
  2. 模块级单例 mysql_manager，import 即可用，无需重复初始化连接池。
  3. 所有方法都做了异常收敛：查询失败返回空结果/None，不抛异常给上层 Agent。
================================================================================
"""

from dbutils.pooled_db import PooledDB
import pymysql
from utils.config_handler import db_conf
from utils.logger_handler import logger

# 从配置中取出 MySQL 参数
_mysql_conf = db_conf["mysql"]


class MySQLManager:
    """MySQL 连接池管理器：对外暴露查询、插入、更新等通用方法。"""

    def __init__(self):
        # 创建连接池：复用 TCP 连接，避免每次操作都重新握手
        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=int(_mysql_conf["pool_size"]),
            host=_mysql_conf["host"],
            port=int(_mysql_conf["port"]),
            user=_mysql_conf["user"],
            password=str(_mysql_conf["password"]),
            database=_mysql_conf["database"],
            charset=_mysql_conf["charset"],
            autocommit=bool(_mysql_conf.get("autocommit", True)),
            cursorclass=pymysql.cursors.DictCursor,  # 查询结果以字典形式返回
        )
        logger.info("[MySQL] 连接池创建成功")

    def _get_conn(self):
        """从连接池中获取一个连接，用完由 with 自动归还。"""
        return self._pool.connection()

    def query(self, sql: str, args: tuple = None) -> list[dict]:
        """
        执行 SELECT 查询，返回字典列表。
        失败时返回空列表，不抛异常。
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, args)
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"[MySQL] 查询失败: {sql} -> {type(e).__name__}: {e}")
            return []

    def query_one(self, sql: str, args: tuple = None) -> dict | None:
        """
        执行 SELECT 查询，返回单条记录（字典）。
        查不到或出错时返回 None。
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, args)
                    return cursor.fetchone()
        except Exception as e:
            logger.error(f"[MySQL] 查询单条失败: {sql} -> {type(e).__name__}: {e}")
            return None

    def execute(self, sql: str, args: tuple = None) -> int:
        """
        执行 INSERT / UPDATE / DELETE，返回受影响行数。
        失败时返回 0。
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    rows = cursor.execute(sql, args)
                    return rows
        except Exception as e:
            logger.error(f"[MySQL] 执行失败: {sql} -> {type(e).__name__}: {e}")
            return 0

    def execute_many(self, sql: str, args_list: list[tuple]) -> int:
        """
        批量执行 INSERT / UPDATE / DELETE，返回受影响总行数。
        适用于一次性导入大量数据。
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cursor:
                    rows = cursor.executemany(sql, args_list)
                    return rows
        except Exception as e:
            logger.error(f"[MySQL] 批量执行失败: {sql} -> {type(e).__name__}: {e}")
            return 0

    def create_database_if_not_exists(self, db_name: str):
        """
        在 MySQL 服务上创建数据库（如果不存在）。
        需要连接到 MySQL 服务器（不指定具体库），单独建一次连接。
        """
        try:
            conn = pymysql.connect(
                host=_mysql_conf["host"],
                port=int(_mysql_conf["port"]),
                user=_mysql_conf["user"],
                password=str(_mysql_conf["password"]),
                charset=_mysql_conf["charset"],
            )
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            conn.close()
            logger.info(f"[MySQL] 数据库 '{db_name}' 确保存在")
        except Exception as e:
            logger.error(f"[MySQL] 创建数据库失败: {type(e).__name__}: {e}")
            raise


# 模块级单例：import 时即创建连接池
# 如果 MySQL 未启动，import 会失败——这是有意为之，让问题早暴露
mysql_manager = MySQLManager()
