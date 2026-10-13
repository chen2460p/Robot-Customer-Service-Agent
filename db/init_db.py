"""
================================================================================
文件：db/init_db.py
职责：数据库初始化入口——建库、建表、从 records.csv 迁移数据到 MySQL。
用法：python db/init_db.py  （首次部署时运行一次即可）

执行流程：
    1. 在 MySQL 服务上创建 robot_customer_service 数据库（如果不存在）
    2. 创建 user_usage_records 表（存储用户使用记录）
    3. 扫描 data/external/records.csv 并批量导入
    4. 打印迁移统计（总记录数、各用户条数等）
================================================================================
"""

import os
import sys

# 确保项目根目录在 sys.path 中，这样 import db.mysql_manager 等才能找到
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.mysql_manager import mysql_manager  # noqa: E402
from utils.logger_handler import logger  # noqa: E402
from utils.path_tool import get_abs_path  # noqa: E402
from utils.config_handler import db_conf  # noqa: E402

# 建表 SQL：user_usage_records 存储用户使用记录
# 字段对应 records.csv 的 6 列：用户ID、特征、清洁效率、耗材、对比、时间
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `user_usage_records` (
    `id`          INT AUTO_INCREMENT PRIMARY KEY,
    `user_id`     VARCHAR(16)  NOT NULL COMMENT '用户ID',
    `feature`     TEXT         NOT NULL COMMENT '清洁特征（户型/人数/地面等）',
    `efficiency`  TEXT         NOT NULL COMMENT '清洁效率',
    `consumables` TEXT         NOT NULL COMMENT '耗材状态',
    `comparison`  TEXT         NOT NULL COMMENT '对比建议',
    `month`       VARCHAR(7)   NOT NULL COMMENT '使用月份，格式 YYYY-MM',
    `created_at`  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_user_month` (`user_id`, `month`)  -- 同一用户同一月份只有一条记录
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# 批量插入 SQL：使用 ON DUPLICATE KEY UPDATE 实现幂等迁移
# 已存在的记录会被更新（以 CSV 数据为准）
INSERT_SQL = """
INSERT INTO user_usage_records (user_id, feature, efficiency, consumables, comparison, month)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    feature    = VALUES(feature),
    efficiency = VALUES(efficiency),
    consumables= VALUES(consumables),
    comparison = VALUES(comparison);
"""


def read_csv(path: str) -> list[tuple]:
    """
    读取 records.csv，跳过表头，返回 (user_id, feature, efficiency, consumables, comparison, month) 元组列表。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:  # 跳过表头
            line = line.strip()
            if not line:
                continue
            arr = [x.replace('"', "") for x in line.split(",")]
            if len(arr) != 6:
                logger.warning(f"[init_db] CSV 行格式异常，跳过: {line[:60]}...")
                continue
            records.append((arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]))

    return records


def init():
    """完整初始化流程：建库 → 建表 → 导入数据 → 打印统计。"""
    # 1. 建库
    db_name = db_conf["mysql"]["database"]
    logger.info(f"[init_db] 步骤 1/4: 确保数据库 '{db_name}' 存在...")
    mysql_manager.create_database_if_not_exists(db_name)

    # 2. 建表
    logger.info("[init_db] 步骤 2/4: 创建 user_usage_records 表...")
    rows = mysql_manager.execute(CREATE_TABLE_SQL)
    logger.info(f"[init_db] 建表完成，受影响行数: {rows}")

    # 3. 读取 CSV
    csv_path = get_abs_path("data/external/records.csv")
    logger.info(f"[init_db] 步骤 3/4: 读取 CSV 文件 {csv_path}...")
    csv_records = read_csv(csv_path)
    logger.info(f"[init_db] CSV 共 {len(csv_records)} 条记录")

    # 4. 批量导入
    logger.info("[init_db] 步骤 4/4: 批量导入 MySQL（ON DUPLICATE KEY UPDATE）...")
    imported = mysql_manager.execute_many(INSERT_SQL, csv_records)
    logger.info(f"[init_db] 导入完成，受影响行数: {imported}")

    # 5. 打印统计
    logger.info("[init_db] ===== 迁移统计 =====")
    total = mysql_manager.query_one("SELECT COUNT(*) AS cnt FROM user_usage_records")
    users = mysql_manager.query_one("SELECT COUNT(DISTINCT user_id) AS cnt FROM user_usage_records")
    months = mysql_manager.query_one("SELECT COUNT(DISTINCT month) AS cnt FROM user_usage_records")
    logger.info(f"  总记录数: {total['cnt']}")
    logger.info(f"  用户数:   {users['cnt']}")
    logger.info(f"  月份数:   {months['cnt']}")
    logger.info("[init_db] 全部完成 ✅")


if __name__ == '__main__':
    try:
        init()
    except Exception as e:
        logger.error(f"[init_db] 初始化失败: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)
