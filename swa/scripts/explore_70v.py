"""
探索 70V 波形特征：找出好波形和坏波形的差异。
"""
import os
import sqlite3
import json
import numpy as np
from scipy import stats as sp_stats

project_dir = r"d:\project\work\swa\swa\src\data\projects\new"
db_path = os.path.join(project_dir, "data.db")
PYTHON = r"d:\project\work\swa\swa\.pixi\envs\default\python.exe"

conn = sqlite3.connect(db_path)
cur = conn.cursor()


def get_table_schema():
    """获取数据库表结构"""
    print("=" * 70)
    print("数据库表结构")
    print("=" * 70)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    for t in tables:
        name = t[0]
        cur.execute(f"PRAGMA table_info({name})")
        cols = cur.fetchall()
        print(f"\n=== {name} ===")
        for c in cols:
            print(f"  {c}")
    print()


def get_record_15890():
    """步骤1: 查看 ID 15890 的完整记录"""
    print("=" * 70)
    print("步骤1: 记录 ID=15890 的完整信息")
    print("=" * 70)

    # 先查看所有列
    cur.execute("PRAGMA table_info(records)")
    record_cols = [c[1] for c in cur.fetchall()]
    print(f"records 表所有列: {record_cols}")

    cur.execute("PRAGMA table_info(waveforms)")
    wave_cols = [c[1] for c in cur.fetchall()]
    print(f"waveforms 表所有列: {wave_cols}")

    # 查询完整记录
    cols_str = ", ".join([f"r.{c}" for c in record_cols])
    query = f"""
        SELECT {cols_str}, w.wave_data
        FROM records r
        JOIN waveforms w ON w.record_id = r.id
        WHERE r.id = 15890
    """
    cur.execute(query)
    row = cur.fetchone()

    if not row:
        print("未找到 record_id=15890 的记录")
        return None

    print("\n完整记录:")
    for i, col in enumerate(record_cols + ["wave_data"]):
        val = row[i]
        if col == "wave_data" and val:
            val_preview = str(val[:120]) + "..." if len(str(val)) > 120 else val
        else:
            val_preview = val
        print(f"  {col}: {val_preview}")

    # 解析波形数据
    wave_data = row[-1]
    if wave_data:
        if isinstance(wave_data, str):
            # 波形数据可能是 JSON 数组或逗号分隔的数值字符串
            if wave_data.strip().startswith("["):
                wave_data = json.loads(wave_data)
            else:
                wave_data = [float(x.strip()) for x in wave_data.split(",") if x.strip()]
        wave_arr = np.array(wave_data, dtype=float)

        print(f"\n波形基本信息:")
        print(f"  数据长度: {len(wave_arr)}")
        print(f"  均值: {np.mean(wave_arr):.4f}")
        print(f"  标准差: {np.std(wave_arr):.4f}")
        print(f"  最大值: {np.max(wave_arr):.4f}")
        print(f"  最小值: {np.min(wave_arr):.4f}")
        print(f"  偏度: {sp_stats.skew(wave_arr):.4f}")
        print(f"  峰度: {sp_stats.kurtosis(wave_arr):.4f}")

        # 计算 FFT 特征
        harmonic_features = compute_fft_features(wave_arr)
        print(f"\nFFT 特征 (前10个谐波幅值):")
        for i, v in enumerate(harmonic_features, 1):
            print(f"  harm_a{i}: {v:.6f}")

        # 计算 alpha_7 和 score
        alpha_7 = harmonic_features[6]  # 第7个谐波 (索引6)
        harm_2_10 = harmonic_features[1:]  # harm_a2 到 harm_a10
        score = alpha_7 / (np.sum(harm_2_10) + 1e-10) if np.sum(harm_2_10) > 0 else 0
        print(f"\n  alpha_7 (harm_a7): {alpha_7:.6f}")
        print(f"  score (a7 / sum(a2..a10)): {score:.6f}")

        # 提取其它字段
        record_data = {}
        for i, col in enumerate(record_cols):
            record_data[col] = row[i]
        record_data["wave_stats"] = {
            "mean": float(np.mean(wave_arr)),
            "std": float(np.std(wave_arr)),
            "max": float(np.max(wave_arr)),
            "min": float(np.min(wave_arr)),
            "skew": float(sp_stats.skew(wave_arr)),
            "kurtosis": float(sp_stats.kurtosis(wave_arr)),
        }
        record_data["harmonics"] = {f"harm_a{i+1}": float(v) for i, v in enumerate(harmonic_features)}
        record_data["alpha_7"] = float(alpha_7)
        record_data["score"] = float(score)

        print()
        return record_data
    return None


def compute_fft_features(wave_arr, n_harmonics=10):
    """计算波形的 FFT 特征 (前 n_harmonics 个谐波幅值)"""
    N = len(wave_arr)
    fft_vals = np.fft.fft(wave_arr)
    # 取幅值 / N
    amplitudes = np.abs(fft_vals[:N//2]) / N
    # 取前 n_harmonics 个谐波 (排除 DC 分量)
    # 基频对应索引1，第i谐波对应索引i
    harmonics = np.zeros(n_harmonics)
    for i in range(n_harmonics):
        idx = i + 1  # 跳过 DC
        if idx < len(amplitudes):
            harmonics[i] = amplitudes[idx]
    return harmonics


def get_70v_records():
    """步骤2: 找出 70V 的记录"""
    print("=" * 70)
    print("步骤2: 查找 70V 记录")
    print("=" * 70)

    # 先查看所有列
    cur.execute("PRAGMA table_info(records)")
    record_cols = [c[1] for c in cur.fetchall()]
    print(f"records 表所有列: {record_cols}")

    # 查看有哪些字段可用于判断预测结果
    predicted_cols = [c for c in record_cols if "predicted" in c.lower()]
    print(f"预测相关列: {predicted_cols}")

    # 找 actual_voltage=70 且 enabled=1 的记录
    query_70v = """
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm, r.device_id
        FROM records r
        WHERE r.actual_voltage = 70 AND r.enabled = 1
        LIMIT 20
    """
    cur.execute(query_70v)
    rows_70v = cur.fetchall()
    print(f"\n找到 {len(rows_70v)} 条 actual_voltage=70 的记录")

    # 如果有 predicted_voltage_1 字段
    if "predicted_voltage_1" in record_cols:
        # 找预测正确 (接近70V) 和错误 (80V以上) 的
        query_good = """
            SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm, r.device_id,
                   r.predicted_voltage_1
            FROM records r
            WHERE r.actual_voltage = 70 AND r.enabled = 1
              AND ABS(r.predicted_voltage_1 - 70) < 5
            LIMIT 5
        """
        query_bad = """
            SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm, r.device_id,
                   r.predicted_voltage_1
            FROM records r
            WHERE r.actual_voltage = 70 AND r.enabled = 1
              AND r.predicted_voltage_1 > 80
            LIMIT 5
        """
        cur.execute(query_good)
        good_rows = cur.fetchall()
        cur.execute(query_bad)
        bad_rows = cur.fetchall()

        ids_good = [r[0] for r in good_rows]
        ids_bad = [r[0] for r in bad_rows]

        print(f"\n预测正确的记录 (predicted ≈ 70V): {len(good_rows)} 条")
        for r in good_rows:
            print(f"  id={r[0]}, actual={r[1]}, temp={r[2]}, hum={r[3]}, rpm={r[4]}, device={r[5]}, predicted={r[6]}")

        print(f"\n预测错误的记录 (predicted > 80V): {len(bad_rows)} 条")
        for r in bad_rows:
            print(f"  id={r[0]}, actual={r[1]}, temp={r[2]}, hum={r[3]}, rpm={r[4]}, device={r[5]}, predicted={r[6]}")
    else:
        # 没有预测字段，按某种规则划分
        print("没有 predicted_voltage_1 字段，随机取几条 70V 记录")
        # 简单取前10条，计算特征后人工判断
        ids_good = []
        ids_bad = []
        for r in rows_70v[:10]:
            print(f"  id={r[0]}, actual={r[1]}, temp={r[2]}, hum={r[3]}, rpm={r[4]}, device={r[5]}")
        ids_good = [r[0] for r in rows_70v[:3]]
        ids_bad = [r[0] for r in rows_70v[3:6]] if len(rows_70v) >= 6 else [r[0] for r in rows_70v[3:]]

    print()
    return ids_good, ids_bad


def compute_wave_features(record_ids, label=""):
    """计算一组记录的特征"""
    print(f"\n{'=' * 70}")
    print(f"波形特征对比: {label}")
    print(f"{'=' * 70}")

    for rid in record_ids:
        cols_str = ", ".join([f"r.{c}" for c in record_cols])
        query = f"""
            SELECT {cols_str}, w.wave_data
            FROM records r
            JOIN waveforms w ON w.record_id = r.id
            WHERE r.id = ?
        """
        cur.execute(query, (rid,))
        row = cur.fetchone()

        if not row:
            print(f"\n--- ID={rid}: 未找到 ---")
            continue

        print(f"\n--- ID={rid} ---")

        # 打印常规字段
        for i, col in enumerate(record_cols):
            val = row[i]
            # 只打印关键字段
            if col in ("id", "actual_voltage", "temperature", "humidity", "rpm", "device_id",
                       "predicted_voltage_1", "predicted_voltage_2", "score"):
                print(f"  {col}: {val}")

        # 解析波形
        wave_data = row[-1]
        if wave_data:
            if isinstance(wave_data, str):
                # 波形数据可能是 JSON 数组或逗号分隔的数值字符串
                if wave_data.strip().startswith("["):
                    wave_data = json.loads(wave_data)
                else:
                    wave_data = [float(x.strip()) for x in wave_data.split(",") if x.strip()]
            wave_arr = np.array(wave_data, dtype=float)

            # 基本统计量
            mean_v = np.mean(wave_arr)
            std_v = np.std(wave_arr)
            max_v = np.max(wave_arr)
            min_v = np.min(wave_arr)
            skew_v = sp_stats.skew(wave_arr)
            kurt_v = sp_stats.kurtosis(wave_arr)
            print(f"  统计量: mean={mean_v:.4f}, std={std_v:.4f}, max={max_v:.4f}, min={min_v:.4f}")
            print(f"  统计量: skew={skew_v:.4f}, kurtosis={kurt_v:.4f}")

            # FFT 特征
            harmonics = compute_fft_features(wave_arr)
            harm_str = ", ".join([f"{v:.6f}" for v in harmonics[:10]])
            print(f"  FFT谐波: [{harm_str}]")

            # alpha_7 和 score
            alpha_7 = harmonics[6]
            harm_2_10 = harmonics[1:]
            score = alpha_7 / (np.sum(harm_2_10) + 1e-10) if np.sum(harm_2_10) > 0 else 0
            print(f"  alpha_7: {alpha_7:.6f}, score(a7/sum(a2..a10)): {score:.6f}")


# ========== 主流程 ==========

# 获取表结构
get_table_schema()

# 步骤1: 查看 ID 15890
rec_15890 = get_record_15890()

# 步骤2: 获取 records 表列名
cur.execute("PRAGMA table_info(records)")
record_cols = [c[1] for c in cur.fetchall()]

# 查找 70V 记录
ids_good, ids_bad = get_70v_records()

# 步骤3: 对比分析
if ids_good and ids_bad:
    compute_wave_features(ids_good, "好波形 (预测正确的 70V)")
    compute_wave_features(ids_bad, "坏波形 (预测错误的 70V)")
else:
    # 如果没有 predicted 字段，直接分析所有 70V 记录
    print("\n没有 predicted_voltage_1 字段，直接分析 70V 记录的特征")
    cur.execute("""
        SELECT r.id FROM records r
        WHERE r.actual_voltage = 70 AND r.enabled = 1
        LIMIT 8
    """)
    all_ids = [r[0] for r in cur.fetchall()]
    compute_wave_features(all_ids[:4], "70V 记录 (前4条)")
    compute_wave_features(all_ids[4:8], "70V 记录 (后4条)")

# === 对比汇总 ===
print("\n" + "=" * 70)
print("对比汇总：好波形 vs 坏波形 关键指标平均值")
print("=" * 70)

def get_features_batch(ids_list):
    """批量获取特征并计算平均值"""
    stats_list = {"mean": [], "std": [], "max": [], "min": [], "skew": [], "kurtosis": []}
    harm_list = {f"harm_a{i+1}": [] for i in range(10)}
    alpha_7_list = []
    score_list = []
    temp_list = []
    hum_list = []
    predicted_list = []

    for rid in ids_list:
        cols_str = ", ".join([f"r.{c}" for c in record_cols])
        query = f"""
            SELECT {cols_str}, w.wave_data
            FROM records r
            JOIN waveforms w ON w.record_id = r.id
            WHERE r.id = ?
        """
        cur.execute(query, (rid,))
        row = cur.fetchone()
        if not row:
            continue

        # 环境参数
        for i, col in enumerate(record_cols):
            if col == "temperature":
                temp_list.append(row[i])
            elif col == "humidity":
                hum_list.append(row[i])
            elif col == "predicted_voltage_1":
                if row[i] is not None:
                    predicted_list.append(row[i])

        wave_data = row[-1]
        if wave_data:
            if isinstance(wave_data, str):
                if wave_data.strip().startswith("["):
                    wave_data = json.loads(wave_data)
                else:
                    wave_data = [float(x.strip()) for x in wave_data.split(",") if x.strip()]
            wave_arr = np.array(wave_data, dtype=float)

            stats_list["mean"].append(np.mean(wave_arr))
            stats_list["std"].append(np.std(wave_arr))
            stats_list["max"].append(np.max(wave_arr))
            stats_list["min"].append(np.min(wave_arr))
            stats_list["skew"].append(sp_stats.skew(wave_arr))
            stats_list["kurtosis"].append(sp_stats.kurtosis(wave_arr))

            harmonics = compute_fft_features(wave_arr)
            for i in range(10):
                harm_list[f"harm_a{i+1}"].append(harmonics[i])

            alpha_7_list.append(harmonics[6])
            harm_2_10 = harmonics[1:]
            score_val = harmonics[6] / (np.sum(harm_2_10) + 1e-10) if np.sum(harm_2_10) > 0 else 0
            score_list.append(score_val)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    result = {}
    for k in stats_list:
        result[k] = avg(stats_list[k])
    for k in harm_list:
        result[k] = avg(harm_list[k])
    result["alpha_7"] = avg(alpha_7_list)
    result["score"] = avg(score_list)
    result["temperature"] = avg(temp_list)
    result["humidity"] = avg(hum_list)
    result["predicted_voltage_1"] = avg(predicted_list)
    result["count"] = len(ids_list)
    return result

good_stats = get_features_batch(ids_good) if ids_good else {}
bad_stats = get_features_batch(ids_bad) if ids_bad else {}

if good_stats and bad_stats:
    print(f"\n{'指标':<20} {'好波形(均值)':<20} {'坏波形(均值)':<20} {'差异':<20}")
    print("-" * 80)

    keys = ["count", "temperature", "humidity", "predicted_voltage_1",
            "mean", "std", "max", "min", "skew", "kurtosis",
            "harm_a1", "harm_a2", "harm_a3", "harm_a4", "harm_a5",
            "harm_a6", "harm_a7", "harm_a8", "harm_a9", "harm_a10",
            "alpha_7", "score"]
    for k in keys:
        if k in good_stats and k in bad_stats:
            diff = good_stats[k] - bad_stats[k]
            print(f"{k:<20} {good_stats[k]:<20.4f} {bad_stats[k]:<20.4f} {diff:<+20.4f}")

    print("\n" + "=" * 70)
    print("关键发现：")
    print("=" * 70)

    # 分析差异
    std_ratio = good_stats["std"] / bad_stats["std"] if bad_stats["std"] else 0
    a7_ratio = good_stats["alpha_7"] / bad_stats["alpha_7"] if bad_stats["alpha_7"] else 0
    print(f"1. 波形标准差(std)：好波形={good_stats['std']:.4f}, 坏波形={bad_stats['std']:.4f}, "
          f"比例={std_ratio:.2f}")
    print(f"   → 坏波形的波动幅度明显更{'大' if std_ratio < 1 else '小'}")

    print(f"2. alpha_7 (第7谐波)：好波形={good_stats['alpha_7']:.4f}, 坏波形={bad_stats['alpha_7']:.4f}, "
          f"比例={a7_ratio:.2f}")
    print(f"   → 坏波形的第7谐波幅值明显更{'高' if a7_ratio < 1 else '低'}")

    print(f"3. Score (a7/sum(a2..a10))：好波形={good_stats['score']:.4f}, 坏波形={bad_stats['score']:.4f}")

    print(f"4. 温湿度差异：temp Δ={good_stats['temperature']-bad_stats['temperature']:+.2f}°C, "
          f"hum Δ={good_stats['humidity']-bad_stats['humidity']:+.1f}%")

conn.close()
print("\n分析完成!")
