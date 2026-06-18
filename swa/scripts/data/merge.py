"""
将数据集内相邻 N 条同温记录的波形拼接（concatenate）为一条长波形。

与平均不同，拼接保留所有原始信息，且没有相位对齐问题。
对拼接后的长波形做拟合时，A1/A3/A5 的精度会更高（更多采样点）。

用法:
    uv run python scripts/data/merge.py --n 4
    uv run python scripts/data/merge.py --n 4 --name default
"""

import sys, os, json, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from lib.utils import dataset_path, dataset_dir


def main():
    parser = argparse.ArgumentParser(description="同温波形拼接合并")
    parser.add_argument("--n", type=int, required=True, help="N 条拼接为 1 条")
    parser.add_argument("--name", default="default", help="数据集名称 (默认 default)")
    parser.add_argument("--temp-field", default="RTU_REGS_P00_ENV_TEMP")
    parser.add_argument("--humid-field", default="RTU_REGS_P00_ENV_HUMIDITY")
    parser.add_argument("--rpm-field", default="RTU_REGS_P00_ROTOR_RPM")
    parser.add_argument("--wave-field", default="RTU_REGS_P00_WAVE_DATA")
    args = parser.parse_args()

    N = args.n
    input_path = dataset_path(args.name)
    output_dir = dataset_dir(f"{args.name}_{N}merge")
    output_path = os.path.join(output_dir, f"{args.name}_{N}merge.jsonl")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"输入: {input_path} ({len(records)} 条)")

    temp_f = args.temp_field
    humid_f = args.humid_field
    rpm_f = args.rpm_field
    wave_f = args.wave_field

    merged = []
    discarded = 0
    buf = []

    def flush_buffer():
        nonlocal discarded
        if len(buf) < N:
            discarded += len(buf)
            buf.clear()
            return

        # 拼接波形
        waves = [b[1] for b in buf]
        wave_concat = np.concatenate(waves)  # (N*512,)

        # 取第一条记录，更新波形字段
        avg_rec = copy.deepcopy(buf[0][0])
        avg_rec[wave_f] = ",".join(f"{v:.6f}" for v in wave_concat)

        # 数值字段取平均
        for field in [temp_f, humid_f]:
            vals = []
            for b in buf:
                v = b[0].get(field)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if vals:
                avg_rec[field] = f"{np.mean(vals):.1f}"

        rpm_vals = []
        for b in buf:
            v = b[0].get(rpm_f)
            if v is not None:
                try:
                    rpm_vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if rpm_vals:
            avg_rec[rpm_f] = f"{np.mean(rpm_vals):.1f}"

        merged.append(avg_rec)
        buf.clear()

    # 逐条处理
    for rec in records:
        wave_str = rec.get(wave_f, "")
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except:
            discarded += 1
            continue

        cur_temp = rec.get(temp_f)

        if not buf:
            buf.append((rec, wave))
            continue

        prev_temp = buf[0][0].get(temp_f)

        if cur_temp == prev_temp or (cur_temp is None and prev_temp is None):
            buf.append((rec, wave))
            if len(buf) == N:
                flush_buffer()
        else:
            flush_buffer()
            buf.append((rec, wave))

    flush_buffer()

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"输出: {output_path} ({len(merged)} 条)")
    print(f"丢弃: {discarded} 条 (未凑齐 {N} 条同温记录)")
    print(f"压缩比: {len(records)} → {len(merged)} ({len(merged)/len(records)*100:.1f}%)")
    print(f"波形长度: {N}×512 = {N*512} 点/条")


if __name__ == "__main__":
    main()
