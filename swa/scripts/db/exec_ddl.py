"""执行 rebuild_tables.sql 重建达梦三张 YSH_OID 表"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import dmPython

conn = dmPython.connect(user="SYSDBA", password="Yshed$888", server="10.21.1.190", port=5236, autoCommit=True)
cur = conn.cursor()
sql = open(os.path.join(os.path.dirname(__file__), "rebuild_tables.sql"), encoding="utf-8").read()
for stmt in sql.split(";"):
    s = stmt.strip()
    if s:
        cur.execute(s)
        print(f"  OK: {s[:70]}...")
conn.close()
print("完成!")
