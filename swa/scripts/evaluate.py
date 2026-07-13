"""
评估模型 — 对数据集进行预测并输出统计和图表。

用法:
    pixi run evaluate --input data/cleaned/all_cleaned.jsonl --model data/models/model.json
    pixi run evaluate --input data/cleaned/all_cleaned.jsonl --model data/models/model.json --plot
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import numpy as np
import matplotlib.pyplot as plt

from swa.data import load_jsonl, parse_wave, clean_voltage_column
from swa.core import VoltagePredictor


@click.command()
@click.option("--input", "-i", required=True, help="输入 JSONL 文件路径")
@click.option("--model", "-m", required=True, help="模型文件路径")
@click.option("--plot", is_flag=True, default=False, help="是否出图")
@click.option("--output", "-o", default=None, help="图表保存路径")
def main(input, model, plot, output):
    # 加载模型
    predictor = VoltagePredictor()
    predictor.load(model)
    click.echo(f"模型: mode={predictor.mode}, window={predictor.window_size}")
    click.echo(f"校准: a={predictor.calib.a:.4f}, b={predictor.calib.b:.2f}")

    # 加载数据
    df = load_jsonl(input)
    df = clean_voltage_column(df)
    click.echo(f"数据: {len(df)} 条")

    # 批量预测
    waves = []
    voltages = []
    humidities = []
    for _, row in df.iterrows():
        wv = parse_wave(row.get("wave_data", "") or "")
        if wv is None:
            continue
        waves.append(wv)
        voltages.append(row["actual_voltage"])

        h = row.get("humidity")
        try:
            humidities.append(float(h) if h else 50.0)
        except (ValueError, TypeError):
            humidities.append(50.0)

    predictions = predictor.predict_batch(waves, humidities)
    valid = [(p, v) for p, v in zip(predictions, voltages) if p is not None]

    if not valid:
        click.echo("无有效预测结果")
        return

    preds, actuals = zip(*valid)
    preds = np.array(preds)
    actuals = np.array(actuals)
    errors = preds - actuals

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2 = float(1 - np.sum(errors ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

    click.echo(f"\n评估结果:")
    click.echo(f"  有效样本: {len(valid)}")
    click.echo(f"  MAE:  {mae:.2f} V")
    click.echo(f"  RMSE: {rmse:.2f} V")
    click.echo(f"  R²:   {r2:.4f}")

    # 分电压统计
    click.echo(f"\n分电压 MAE:")
    unique_voltages = sorted(set(actuals))
    for v in unique_voltages:
        mask = actuals == v
        if np.sum(mask) > 0:
            v_mae = float(np.mean(np.abs(errors[mask])))
            cnt = int(np.sum(mask))
            click.echo(f"  {v:+.0f}V: {v_mae:.2f}V (n={cnt})")

    # 绘图
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 预测 vs 实际
        ax = axes[0]
        ax.scatter(actuals, preds, alpha=0.5, s=10)
        vmin, vmax = min(actuals), max(actuals)
        ax.plot([vmin, vmax], [vmin, vmax], "r--", alpha=0.5)
        ax.set_xlabel("实际电压 (V)")
        ax.set_ylabel("预测电压 (V)")
        ax.set_title(f"预测 vs 实际 (MAE={mae:.2f}V)")
        ax.grid(True, alpha=0.3)

        # 误差分布
        ax = axes[1]
        ax.hist(errors, bins=50, alpha=0.7, color="#0078d4")
        ax.set_xlabel("误差 (V)")
        ax.set_ylabel("频次")
        ax.set_title(f"误差分布 (RMSE={rmse:.2f}V)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if output:
            plt.savefig(output, dpi=150)
            click.echo(f"图表已保存: {output}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
