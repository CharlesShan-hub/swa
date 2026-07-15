"""
最小二乘法电压检测

利用最小二乘周期投影提取波形特征，归一化后用线性回归拟合电压绝对值。

特征（原始 → 归一化）:
  - alpha_7:      7.0 周期余弦分量幅值（对湿度和 RPM 不敏感）
  - score:        7.0 + 8.1 周期加权评分
  - harm_a1:      FFT 基波幅值
  - temperature:  环境温度（z-score 归一化）
  - humidity:     环境湿度（z-score 归一化）
  - rpm:          转子转速（z-score 归一化）

归一化:
  在训练集上计算各特征的均值/标准差，同步应用到测试集，
  使回归系数反映各特征的真实重要性。

滑动窗口:
  对同一电压等级内按时间排序的连续记录做窗口平均。
  窗口大小 > 1 时，窗口内各记录的指标取平均作为一条训练/测试样本。
  例: 70V 有 100 条，窗口 16 → 1-16, 2-17, ..., 85-100 共 85 个窗口。

方法:
  1. 对每条波形提取特征
  2. (可选) 滑动窗口平均
  3. 特征归一化（z-score）
  4. 最小二乘线性回归: |voltage| = w0 + w1*z1 + w2*z2 + ...
  5. 在测试集上预测并评估（仅预测绝对值，不区分正负）"""

from typing import Optional
import os
import sqlite3

import numpy as np
import pandas as pd

from swa.core.scoring import compute_score, compute_alpha7

name = "最小二乘法 (LS)"

# ── 特征提取 ────────────────────────────────────────────────────


def _extract_features(wave: np.ndarray) -> dict[str, float]:
    """从波形提取特征向量。"""
    features = {}

    # 最小二乘投影特征
    alpha7 = compute_alpha7(wave)
    features["alpha_7"] = alpha7 if alpha7 is not None else 0.0

    score = compute_score(wave)
    features["score"] = score

    # FFT 特征（仅 harm_a1）
    y = wave - np.mean(wave)
    n = len(y)
    if n >= 20:
        fft_vals = np.fft.rfft(y)
        mag = np.abs(fft_vals[1:])
        if len(mag) > 3:
            search_end = min(len(mag), n // 3)
            fund_idx = int(np.argmax(mag[:search_end]) + 1)
            a1 = float(mag[fund_idx - 1])
            features["harm_a1"] = a1 if a1 > 0 else 0.0
        else:
            features["harm_a1"] = 0.0
    else:
        features["harm_a1"] = 0.0

    return features


_FEATURE_NAMES = ["alpha_7", "score", "harm_a1", "temperature", "humidity", "rpm"]


def _apply_window(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """对每个电压等级内的记录做滑动窗口平均。

    Args:
        df: 包含 actual_voltage 和各特征的 DataFrame，按 id 排序
        window_size: 窗口大小（1 = 不做平均）

    Returns:
        窗口平均后的 DataFrame
    """
    if window_size <= 1:
        return df

    windows = []
    # 按电压分组（已是连续排列）
    for voltage, group in df.groupby("actual_voltage", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < window_size:
            # 记录太少，整个电压作为一个窗口
            row = {"actual_voltage": voltage}
            for feat in _FEATURE_NAMES:
                row[feat] = float(group[feat].mean())
            row["window_ids"] = list(group["id"])
            row["window_count"] = n
            windows.append(row)
            continue

        for start in range(n - window_size + 1):
            seg = group.iloc[start : start + window_size]
            row = {"actual_voltage": voltage}
            for feat in _FEATURE_NAMES:
                row[feat] = float(seg[feat].mean())
            row["window_ids"] = list(seg["id"])
            row["window_count"] = window_size
            windows.append(row)

    return pd.DataFrame(windows)


def _load_data(
    project_dir: str,
    train_voltages: list[float],
    test_voltages: list[float],
    window_size: int = 1,
    max_samples_per_voltage: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载训练集和测试集的波形数据。

    Args:
        project_dir: 项目目录
        train_voltages: 训练电压列表
        test_voltages: 测试电压列表
        window_size: 滑动窗口大小（默认 1 = 不做平均）
        max_samples_per_voltage: 每电压最大样本数（<=0 不限）

    Returns:
        (train_df, test_df), 每列包含 actual_voltage 和各特征
    """
    db_path = os.path.join(project_dir, "data.db")
    conn = sqlite3.connect(db_path)

    # 先查数据库里有哪些电压
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT actual_voltage FROM records ORDER BY actual_voltage")
    all_voltages = [r[0] for r in cur.fetchall()]

    # 匹配用户选择的电压
    def match_voltages(selected: list[float]) -> list[float]:
        matched = []
        for sv in selected:
            for av in all_voltages:
                if abs(av - sv) < 1e-6:
                    matched.append(av)
                    break
        return matched

    train_v = match_voltages(train_voltages)
    test_v = match_voltages(test_voltages)

    # 获取全部数据（按 id 排序以确保时间顺序）
    rows = conn.execute("""
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm, w.wave_data
        FROM records r
        JOIN waveforms w ON w.record_id = r.id
        WHERE r.enabled = 1
        ORDER BY r.id
    """).fetchall()

    conn.close()

    # 提取特征
    records = []
    for row in rows:
        rid, voltage, temp, humid, rpm_val, wave_str = row
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except (ValueError, TypeError, AttributeError):
            continue
        if len(wave) < 20:
            continue

        feats = _extract_features(wave)
        feats["id"] = rid
        feats["actual_voltage"] = voltage
        feats["temperature"] = float(temp) if temp is not None else 0.0
        feats["humidity"] = float(humid) if humid is not None else 0.0
        feats["rpm"] = float(rpm_val) if rpm_val is not None else 0.0
        records.append(feats)

    df = pd.DataFrame(records)

    # 滑动窗口平均
    if window_size > 1:
        df = _apply_window(df, window_size)

    # 每电压最大样本数限制
    df = _limit_per_voltage(df, max_samples_per_voltage)

    train_df = df[df["actual_voltage"].isin(train_v)].copy()
    test_df = df[df["actual_voltage"].isin(test_v)].copy()

    return train_df, test_df


def _limit_per_voltage(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """每个电压等级最多保留 max_samples 条（取前 N 条）。

    Args:
        df: 含 actual_voltage 列的 DataFrame
        max_samples: 最大条数（<=0 时不限）

    Returns:
        采样后的 DataFrame
    """
    if max_samples <= 0:
        return df
    return df.groupby("actual_voltage", sort=False).head(max_samples).reset_index(drop=True)


def _normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """z-score 归一化：在训练集上计算均值/标准差，同步应用到测试集。

    Returns:
        (train_norm, test_norm, norm_params)
        norm_params = {特征名: {"mean": float, "std": float}}
    """
    norm_params = {}
    for col in feature_names:
        vals = train_df[col].values
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma < 1e-12:
            sigma = 1.0
        norm_params[col] = {"mean": mu, "std": sigma}
        train_df[col] = (train_df[col] - mu) / sigma
        test_df[col] = (test_df[col] - mu) / sigma
    return train_df, test_df, norm_params


# ── 运行入口 ────────────────────────────────────────────────────


def run(
    project_dir: str,
    train_voltages: list[float],
    test_voltages: list[float],
    window_size: int = 1,
    max_samples_per_voltage: int = 0,
) -> dict:
    """运行最小二乘法检测。

    Args:
        project_dir: 项目目录路径（含 data.db）
        train_voltages: 用于训练的电压值列表
        test_voltages: 用于测试的电压值列表
        window_size: 滑动窗口大小（默认 1 = 不做平均）
        max_samples_per_voltage: 每电压最多样本数（<=0 不限）

    Returns:
        dict: {
            "metrics": {
                "mae": float,
                "rmse": float,
                "r2": float,
                "mape": float,
                "train_count": int,
                "test_count": int,
            },
            "coefficients": dict[str, float],    # 回归系数
            "intercept": float,                   # 截距
            "train_results": list[dict],           # 各训练样本 (actual, pred)
            "test_results": list[dict],            # 各测试样本 (actual, pred)
            "voltage_mae": dict[str, float],       # 各电压的 MAE
            "window_size": int,                    # 实际使用的窗口大小
        }
    """
    train_df, test_df = _load_data(
        project_dir, train_voltages, test_voltages,
        window_size=window_size,
        max_samples_per_voltage=max_samples_per_voltage,
    )

    if len(train_df) < 5:
        return {"error": f"训练数据不足 ({len(train_df)} 条)，至少需要 5 条"}

    # ── 特征归一化 ──
    train_df, test_df, norm_params = _normalize(train_df, test_df, _FEATURE_NAMES)

    # ── 训练：最小二乘线性回归（目标 = 电压绝对值）──
    X_train = train_df[_FEATURE_NAMES].values
    y_train = np.abs(train_df["actual_voltage"].values)

    # 加截距列
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])

    coeffs, residuals, rank, s = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    intercept = float(coeffs[0])
    weights = {name: float(coeffs[i + 1]) for i, name in enumerate(_FEATURE_NAMES)}

    # ── 预测 ──
    def predict(X: np.ndarray) -> np.ndarray:
        X_aug = np.column_stack([np.ones(len(X)), X])
        return X_aug @ coeffs

    train_pred = predict(X_train)
    test_pred = predict(test_df[_FEATURE_NAMES].values)

    # ── 评估指标（基于绝对值）──
    def calc_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mape = float(np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100)
        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

    train_actual_abs = np.abs(train_df["actual_voltage"].values)
    test_actual_abs = np.abs(test_df["actual_voltage"].values)
    train_metrics = calc_metrics(train_actual_abs, train_pred)
    test_metrics = calc_metrics(test_actual_abs, test_pred)

    # 各电压 MAE（按绝对值分组，+110V 和 -110V 合并）
    voltage_mae = {}
    test_df_abs = test_df.copy()
    test_df_abs["abs_voltage"] = test_df_abs["actual_voltage"].abs()
    for v_abs in sorted(set(test_df_abs["abs_voltage"])):
        mask = test_df_abs["abs_voltage"] == v_abs
        v_actual = test_df_abs.loc[mask, "abs_voltage"].values
        v_pred = test_pred[mask]
        voltage_mae[f"{v_abs:.0f}V"] = float(np.mean(np.abs(v_actual - v_pred)))

    # 构建结果列表（窗口模式用 window_ids，否则用 id）
    use_window = window_size > 1
    train_results = [
        {
            "id": int(row["window_ids"][0]) if use_window else int(row["id"]),
            "ids": row["window_ids"] if use_window else [int(row["id"])],
            "actual": abs(float(row["actual_voltage"])),
            "pred": float(pred),
        }
        for row, pred in zip(train_df.to_dict("records"), train_pred)
    ]
    test_results = [
        {
            "id": int(row["window_ids"][0]) if use_window else int(row["id"]),
            "ids": row["window_ids"] if use_window else [int(row["id"])],
            "actual": abs(float(row["actual_voltage"])),
            "pred": float(pred),
        }
        for row, pred in zip(test_df.to_dict("records"), test_pred)
    ]

    return {
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "train_count": len(train_df),
            "test_count": len(test_df),
        },
        "coefficients": weights,
        "intercept": intercept,
        "train_results": train_results,
        "test_results": test_results,
        "voltage_mae": voltage_mae,
        "window_size": window_size,
        "max_samples_per_voltage": max_samples_per_voltage,
        "norm_params": norm_params,
    }
