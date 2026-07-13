"""
训练/校准电压预测模型。

用法:
    pixi run train --input data/cleaned/all_cleaned.jsonl --mode score
    pixi run train --input data/cleaned/all_cleaned.jsonl --mode alpha7
    pixi run train --input data/cleaned/all_cleaned.jsonl --mode alpha7 --with-humidity
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import numpy as np
from swa.data import load_jsonl, parse_wave, clean_voltage_column, dataset_summary
from swa.core import VoltagePredictor


@click.command()
@click.option("--input", "-i", required=True, help="输入 JSONL 文件路径")
@click.option("--output", "-o", default="data/models/model.json", help="模型输出路径")
@click.option("--mode", default="score", type=click.Choice(["score", "alpha7"]), help="特征模式")
@click.option("--window", type=int, default=20, help="S20 窗口大小")
@click.option("--f1", type=float, default=7.0, help="第一个周期数")
@click.option("--f2", type=float, default=8.1, help="第二个周期数")
@click.option("--w", type=float, default=0.25, help="beta 权重")
@click.option("--with-humidity", is_flag=True, default=False, help="使用湿度补偿")
def main(input, output, mode, window, f1, f2, w, with_humidity):
    # 加载数据
    click.echo(f"加载数据: {input}")
    df = load_jsonl(input)
    df = clean_voltage_column(df)
    click.echo(f"有效数据: {len(df)} 条")

    # 解析波形
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
        if h is not None:
            try:
                humidities.append(float(h))
            except (ValueError, TypeError):
                humidities.append(50.0)
        else:
            humidities.append(50.0)

    click.echo(f"有效波形: {len(waves)} 条")

    # 计算特征
    click.echo(f"计算特征 (mode={mode})...")
    predictor = VoltagePredictor(window_size=window, f1=f1, f2=f2, w=w, mode=mode)
    scores = [predictor.compute_feature(w) for w in waves]
    scores = [s for s in scores if s is not None]

    if len(scores) < 10:
        click.echo("有效特征太少，无法校准")
        return

    # 校准
    if with_humidity and len(humidities) == len(scores):
        click.echo("使用湿度补偿校准...")
        calib = predictor.fit_calibration_with_humidity(scores, voltages[:len(scores)], humidities[:len(scores)])
    else:
        click.echo("使用线性校准...")
        calib = predictor.fit_calibration(scores, voltages[:len(scores)])

    click.echo(f"校准参数: a={calib.a:.4f}, b={calib.b:.2f}", err=True)
    if calib.use_humidity:
        click.echo(f"           c={calib.c:.4f}, h0={calib.h0:.1f}", err=True)

    # 评估
    predictions = [calib.predict(s, h) for s, h in zip(scores, humidities[:len(scores)])]
    errors = np.array(predictions) - np.array(voltages[:len(predictions)])
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    click.echo(f"MAE={mae:.2f}V, RMSE={rmse:.2f}V")

    # 保存
    predictor.save(output)
    click.echo(f"模型已保存: {output}")


if __name__ == "__main__":
    main()
