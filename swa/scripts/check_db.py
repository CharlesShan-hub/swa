"""查达梦数据库数据量"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.swa.db.connection import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
total = cur.fetchone()[0]
print(f"数据库总条数: {total}")

cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ACTUAL_VOLTAGE IS NULL")
null_v = cur.fetchone()[0]
print(f"无标签数据: {null_v}")

cur.execute("SELECT MIN(SYSTEM_TIME), MAX(SYSTEM_TIME) FROM YS_DB.TB_MODBUS_DEV_POINT")
min_t, max_t = cur.fetchone()
print(f"日期范围: {min_t} ~ {max_t}")

cur.execute("""
    SELECT ACTUAL_VOLTAGE, COUNT(*) as cnt 
    FROM YS_DB.TB_MODBUS_DEV_POINT 
    WHERE ACTUAL_VOLTAGE IS NOT NULL
    GROUP BY ACTUAL_VOLTAGE 
    ORDER BY cnt DESC 
    LIMIT 10
""")
print("电压分布(前10):")
for row in cur.fetchall():
    print(f"  {str(row[0]):>10}  {row[1]}条")

conn.close()
