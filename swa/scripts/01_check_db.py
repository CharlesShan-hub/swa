"""
步骤 ①：测试数据库连通性

检查能否连上达梦数据库，并查看表基本信息。

用法：
    uv run python scripts/01_check_db.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.swa.db.connection import get_connection


def main():
    print("=" * 50)
    print("  数据库连通性检查")
    print("=" * 50)

    try:
        conn = get_connection()
        cur = conn.cursor()
        print("  ✅ 数据库连接成功！\n")
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        return

    # 检查表是否存在
    try:
        cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
        total = cur.fetchone()[0]
        print(f"  📊 TB_MODBUS_DEV_POINT 表: {total} 条记录")

        # 查看最近 3 条
        cur.execute("""
            SELECT TEST_CASE_CODE, SYSTEM_TIME, ACTUAL_VOLTAGE
            FROM YS_DB.TB_MODBUS_DEV_POINT
            WHERE ROWNUM <= 3
            ORDER BY SYSTEM_TIME DESC
        """)
        rows = cur.fetchall()
        print(f"\n  最近 3 条记录：")
        for r in rows:
            print(f"    {r[0]:>10} | {r[1]} | {r[2]}")
    except Exception as e:
        print(f"  ❌ 查询失败: {e}")

    conn.close()
    print("\n  ✅ 连接已关闭")


if __name__ == "__main__":
    main()
