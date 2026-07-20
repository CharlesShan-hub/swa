"""
查询数据库某表的一行全部内容，看看有哪些字段。
用法: pixi run python scripts/db/show_row.py [ROWID]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from swa.db.connection import get_connection

rowid = sys.argv[1] if len(sys.argv) > 1 else "1"

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT * FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ROWID = ?", (rowid,))
col_names = [desc[0] for desc in cur.description]
row = cur.fetchone()

if row is None:
    print(f"ROWID={rowid} 未找到")
else:
    print(f"=== ROWID={rowid}  共 {len(col_names)} 列 ===")
    for name, val in zip(col_names, row):
        s = str(val)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"  {name}: {s}")
