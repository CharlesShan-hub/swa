"""
回填 local.jsonl 中的 DEVICE_ID。

原理：
  1. 从数据库按 ROWID 顺序拉取所有 (ROWID, DEVICE_ID, SYSTEM_TIME)
  2. jsonl 也是按 ORDER BY ROWID 下载的，行号一一对应
  3. 用前几条的 SYSTEM_TIME 做交叉校验，确保没错位

用法：pixi run python scripts/db/backfill_device_id.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from swa.db.connection import get_connection

JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "local.jsonl"
)

# ── 1. 从 DB 拉 DEVICE_ID ────────────────────────────────────
print("连接数据库...")
conn = get_connection()
cur = conn.cursor()

print("拉取 DEVICE_ID（按 ROWID 排序）...")
cur.execute("SELECT ROWID, DEVICE_ID, SYSTEM_TIME FROM YS_DB.TB_MODBUS_DEV_POINT ORDER BY ROWID")
db_rows = cur.fetchall()
print(f"  数据库共 {len(db_rows)} 条")
conn.close()

# ── 2. 读 jsonl ──────────────────────────────────────────────
print(f"读取 {JSONL_PATH}...")
with open(JSONL_PATH, encoding="utf-8") as f:
    jsonl_lines = f.readlines()
jsonl_lines = [l for l in jsonl_lines if l.strip()]
print(f"  jsonl 共 {len(jsonl_lines)} 条")

# ── 3. 数量校验：取 jsonl 的实际条数（可能少于 DB，因为 DB 有新数据）─
n = min(len(db_rows), len(jsonl_lines))
if len(db_rows) != len(jsonl_lines):
    print(f"\n📌 数据库比 jsonl 多 {len(db_rows) - len(jsonl_lines)} 条（新数据），对齐前 {n} 条")
db_rows = db_rows[:n]
jsonl_lines = jsonl_lines[:n]

# ── 4. 交叉校验：检查前5条的 SYSTEM_TIME ──────────────────────
print("\n交叉校验前5条...")
match_ok = True
for i in range(min(5, len(db_rows), len(jsonl_lines))):
    line = json.loads(jsonl_lines[i])
    db_time = str(db_rows[i][2]) if db_rows[i][2] else ""
    jl_time = str(line.get("SYSTEM_TIME", ""))
    ok = db_time[:16] == jl_time[:16]  # 只比到分钟
    if not ok:
        print(f"  ⚠️  第{i}行 时间不匹配: DB={db_time}  jsonl={jl_time}")
        match_ok = False
    else:
        print(f"  ✓  第{i}行 {db_time[:19]}")

if not match_ok:
    print("\n❌ 交叉校验失败，可能数据错位！已停止。")
    exit(1)

# ── 5. 回填 ──────────────────────────────────────────────────
print(f"\n回填 DEVICE_ID...")
updated = 0
skipped = 0
new_lines = []
for i, (dbr, line) in enumerate(zip(db_rows, jsonl_lines)):
    rec = json.loads(line)
    db_dev_id = dbr[1]  # DEVICE_ID
    old_dev_id = rec.get("DEVICE_ID")

    if old_dev_id == db_dev_id:
        new_lines.append(line)
        skipped += 1
        continue

    rec["DEVICE_ID"] = db_dev_id
    new_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
    updated += 1

    if updated <= 3 or updated % 5000 == 0:
        print(f"  已处理 {i+1}/{len(db_rows)} 行，更新 {updated} 条")

# ── 6. 写回 ──────────────────────────────────────────────────
print(f"\n写回文件...（更新 {updated} 条，跳过 {skipped} 条）")
with open(JSONL_PATH, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("完成！")
