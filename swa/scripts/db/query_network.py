"""
查询 YS_DB_HD 下三张网络拓扑表的完整结构和数据。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dmPython
from pprint import pprint

conn = dmPython.connect(user="SYSDBA", password="Yshed$888", server="10.21.1.190", port=5236, autoCommit=True)
cur = conn.cursor()

for tbl in ["YSH_OID_DEVICE", "YSH_OID_PORT", "YSH_OID_CONN"]:
    print(f"\n{'='*80}")
    print(f"  {tbl}")
    print(f"{'='*80}")
    
    # 数据 — 直接 SELECT * 看完整列名和内容
    cur.execute(f"SELECT * FROM YS_DB_HD.{tbl}")
    col_names = [desc[0] for desc in cur.description]
    print(f"  列 ({len(col_names)}): {col_names}")
    rows = cur.fetchall()
    print(f"  数据 ({len(rows)} 行):")
    for r in rows[:20]:
        vals = " | ".join(str(v)[:40] if v is not None else "NULL" for v in r)
        print(f"    {vals}")
    if len(rows) > 20:
        print(f"    ... 还有 {len(rows)-20} 行")

conn.close()
