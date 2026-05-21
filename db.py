# db.py
import sqlite3
import os
from datetime import datetime



# 数据库文件路径
DB_PATH = "processing.db"

def get_db_connection():
    """获取数据库连接，并启用外键约束"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """
    初始化数据库：创建 processing_records 表（如果不存在）。
    表字段说明：
    - id: 自增主键
    - input_path: 输入图片的绝对路径（唯一约束，防止重复处理）
    - output_path: 输出图片的路径（可能为空）
    - subject_count: 识别出的主体人物数量
    - stray_count: 路人数量
    - status: 'success' 或 'failed'
    - error_message: 失败时的错误信息
    - elapsed_seconds: 处理耗时（秒）
    - created_at: 记录创建时间（ISO格式字符串）
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_path TEXT UNIQUE NOT NULL,
            output_path TEXT,
            subject_count INTEGER,
            stray_count INTEGER,
            status TEXT CHECK(status IN ('success', 'failed')),
            error_message TEXT,
            elapsed_seconds REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def is_image_processed(input_path: str) -> bool:
    """
    检查某张图片是否已经处理过（且状态为 success）
    返回 True 表示已成功处理，可以跳过
    """
    abs_path = os.path.abspath(input_path)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM processing_records WHERE input_path = ? AND status = 'success'",
        (abs_path,)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def insert_record(input_path: str, output_path: str, subject_count: int, stray_count: int,
                  status: str, error_message: str, elapsed: float):
    """
    插入一条处理记录。
    如果 input_path 已存在（重复），则更新记录（用 REPLACE 或 INSERT OR REPLACE）。
    这里使用 INSERT OR REPLACE 保证唯一性。
    """
    abs_input = os.path.abspath(input_path)
    abs_output = os.path.abspath(output_path) if output_path else None
    created_at = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processing_records 
        (input_path, output_path, subject_count, stray_count, status, error_message, elapsed_seconds, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (abs_input, abs_output, subject_count, stray_count, status, error_message, elapsed, created_at))
    conn.commit()
    conn.close()


def get_failed_records():
    """查询所有处理失败的记录，返回列表，每个元素为字典"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT input_path, error_message, created_at
        FROM processing_records
        WHERE status = 'failed'
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [{"input_path": row[0], "error_message": row[1], "created_at": row[2]} for row in rows]


def get_success_records():
    """查询所有处理成功的记录，返回列表，每个元素为字典"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT input_path, output_path, created_at, subject_count, stray_count
        FROM processing_records
        WHERE status = 'success'
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [{"input_path": row[0], "output_path": row[1], "created_at": row[2], "subject_count": row[3], "stray_count": row[4]} for row in rows]