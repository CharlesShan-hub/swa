"""
验证新湿度校正公式：A1_corrected = A1 - (a×V + b)×(H - 40)

即湿度对 A1 的偏移量与电压成正比。
对比校正前后的 A1 vs 电压曲线是否更平滑（各湿度层应融合为一条线）。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT r.id, r.actual_voltage, r.humidity, r.device_id, r.harm_a1
        FROM records r
        WHERE r.enabled = 1 AND r.actual_voltage >= 0
          AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
        ORDER BY r.id
    """, conn)
    conn.close()
    return df


def fit_humidity_correction(df: pd.DataFrame, dev: str):
    """拟合新校正公式: A1 = α·|V| + k·V·(H-40) + γ·(H-40) + β
    
    展开: A1 = β + α·V + k·V·H' + γ·H'
    其中 H' = H - 40
    """
    sub = df[df["device_id"] == dev].copy()
    sub = sub.dropna(subset=["harm_a1", "actual_voltage", "humidity"])
    if len(sub) < 20:
        return None, None

    V = sub["actual_voltage"].abs().values
    Hp = sub["humidity"].values - 40.0  # H' = H - 40
    y = sub["harm_a1"].values

    # 设计矩阵: [1, V, V*H', H']
    X = np.column_stack([np.ones(len(V)), V, V * Hp, Hp])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta, alpha, k, gamma = coeffs

    print(f"\n  设备 {dev[:20]:20s}")
    print(f"    截距 β = {beta:.4f}")
    print(f"    电压 α = {alpha:.4f}")
    print(f"    交互 k = {k:.6f}  (V×H' 项)")
    print(f"    湿度 γ = {gamma:.4f}  (H' 项)")
    print(f"    校正公式: A1_corr = A1 - ({k:.6f}×V + {gamma:.4f}) × (H - 40)")

    def correct(row):
        return row["harm_a1"] - (k * abs(row["actual_voltage"]) + gamma) * (row["humidity"] - 40.0)

    return correct, {"beta": beta, "alpha": alpha, "k": k, "gamma": gamma}


def evaluate_correction(df: pd.DataFrame, dev: str, correct_fn):
    """对比校正前后各湿度层的 A1~V 曲线是否融合。"""
    sub = df[df["device_id"] == dev].copy()

    # 湿度分箱
    sub["hum_bin"] = pd.cut(sub["humidity"], bins=range(25, 70, 5), right=False)

    # 校正前
    sub["a1_raw"] = sub["harm_a1"]
    # 校正后
    if correct_fn is not None:
        sub["a1_corrected"] = sub.apply(correct_fn, axis=1)
    else:
        sub["a1_corrected"] = sub["a1_raw"]

    # 每个 (电压, 湿度层) 的均值
    stats = sub.groupby(["actual_voltage", "hum_bin"])[["a1_raw", "a1_corrected"]].mean().reset_index()

    # 打印
    print(f"\n  ── 校正前 A1 (按电压×湿度) ──")
    pivot_raw = stats.pivot_table(values="a1_raw", index="hum_bin", columns="actual_voltage", aggfunc="mean")
    print(pivot_raw.to_string(float_format="%.1f"))

    print(f"\n  ── 校正后 A1 (按电压×湿度) ──")
    pivot_corr = stats.pivot_table(values="a1_corrected", index="hum_bin", columns="actual_voltage", aggfunc="mean")
    print(pivot_corr.to_string(float_format="%.1f"))

    # 计算各电压下 A1 的湿度层间标准差（越小说明校正越好）
    raw_std = sub.groupby("actual_voltage")["a1_raw"].std()
    corr_std = sub.groupby("actual_voltage")["a1_corrected"].std()

    # 但更好的指标：各电压内不同湿度层的 A1 均值标准差
    raw_layer_std = stats.groupby("actual_voltage")["a1_raw"].std()
    corr_layer_std = stats.groupby("actual_voltage")["a1_corrected"].std()

    print(f"\n  各电压内湿度层间 A1 标准差（越小越好）:")
    for v in sorted(sub["actual_voltage"].unique()):
        if v in raw_layer_std.index and v in corr_layer_std.index:
            print(f"    V={v:+.0f}  校正前 σ={raw_layer_std[v]:.1f}  校正后 σ={corr_layer_std[v]:.1f}  改善={raw_layer_std[v]-corr_layer_std[v]:.1f}")

    return sub


def plot_comparison(df_dict: dict, out_path: str):
    """每个设备一行，校正前/后两列，画 A1~V 曲线，按湿度分层着色。"""
    devices = list(df_dict.keys())
    n = len(devices)

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, (dev, sub) in enumerate(df_dict.items()):
        sub = sub.copy()
        sub["hum_bin"] = pd.cut(sub["humidity"], bins=range(25, 70, 5), right=False)
        hum_labels = sorted(sub["hum_bin"].dropna().unique(), key=str)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(hum_labels)))

        for col, col_name, a1_col in [(0, "校正前", "a1_raw"), (1, "校正后", "a1_corrected")]:
            ax = axes[row, col]
            for hbin, color in zip(hum_labels, colors):
                mask = sub["hum_bin"] == hbin
                if mask.sum() < 5:
                    continue
                # 按电压取均值
                grp = sub[mask].groupby("actual_voltage")[a1_col].mean()
                ax.plot(grp.index, grp.values, "o-", label=str(hbin), color=color, markersize=3, linewidth=1)

            short_dev = dev[:16] if len(dev) > 16 else dev
            ax.set_title(f"{short_dev}  {col_name}", fontsize=11)
            ax.set_xlabel("电压 (V)")
            ax.set_ylabel("A1")
            ax.legend(fontsize=6, title="湿度", title_fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n已保存: {out_path}")
    plt.close(fig)


def main():
    df = load_data()
    devices = sorted(df["device_id"].dropna().unique())
    print(f"总记录: {len(df)}, 设备数: {len(devices)}")

    df_corrected = df.copy()
    df_dict = {}

    for dev in devices:
        print(f"\n{'=' * 60}")
        print(f"拟合设备: {dev[:24]}")
        print("=" * 60)

        correct_fn, params = fit_humidity_correction(df, dev)
        sub = evaluate_correction(df, dev, correct_fn)

        if correct_fn is not None:
            mask = df_corrected["device_id"] == dev
            df_corrected.loc[mask, "harm_a1"] = df_corrected.loc[mask].apply(correct_fn, axis=1)

        df_dict[dev] = sub

    # 画对比图
    plot_comparison(df_dict, os.path.join(os.path.dirname(__file__), "humidity_correction_v2.png"))


if __name__ == "__main__":
    main()
