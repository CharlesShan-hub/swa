"""
将 exported_data.jsonl 中相邻 N 条同温度记录的波形合并平均。

用法:
    uv run python scripts/merge_waveforms.py --n 4
    uv run python scripts/merge_waveforms.py --n 4 --input data/exported_data.jsonl
"""

import sys, os, json, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="波形合并平均")
    parser.add_argument("--n", type=int, required=True, help="N 条合并为 1 条")
    parser.add_argument("--input", default="data/exported_data.jsonl", help="输入文件")
    parser.add_argument("--temp-field", default="RTU_REGS_P00_ENV_TEMP",
                        help="温度字段名 (默认 RTU_REGS_P00_ENV_TEMP)")
    parser.add_argument("--humid-field", default="RTU_REGS_P00_ENV_HUMIDITY",
                        help="湿度字段名 (默认 RTU_REGS_P00_ENV_HUMIDITY)")
    parser.add_argument("--rpm-field", default="RTU_REGS_P00_ROTOR_RPM",
                        help="转速字段名 (默认 RTU_REGS_P00_ROTOR_RPM)")
    parser.add_argument("--wave-field", default="RTU_REGS_P00_WAVE_DATA",
                        help="波形字段名 (默认 RTU_REGS_P00_WAVE_DATA)")
    args = parser.parse_args()

    N = args.n
    input_path = args.input
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_{N}merge.jsonl"

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
    buf = []  # 缓冲区: [(rec, wave_array), ...]

    def flush_buffer():
        """如果缓冲区不足 N 条，丢弃；否则平均并写入 merged"""
        nonlocal discarded
        if len(buf) < N:
            discarded += len(buf)
            buf.clear()
            return

        # 验证 N 条的温度完全一致（虽然按此逻辑 buf 里 temp 一定相等）
        temps = set(b[0].get(temp_f) for b in buf)
        if len(temps) > 1:
            discarded += len(buf)
            buf.clear()
            return

        # 平均波形（逐点平均）
        waves = np.array([b[1] for b in buf])  # (N, 512)
        wave_avg = waves.mean(axis=0)

        # 取第一条记录，更新数值字段
        avg_rec = copy.deepcopy(buf[0][0])

        # 波形: 替换为平均后的波形
        avg_rec[wave_f] = ",".join(f"{v:.6f}" for v in wave_avg)

        # 数值字段: 取平均
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

        # RPM 取平均
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
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        except:
            discarded += 1
            continue

        cur_temp = rec.get(temp_f)

        # 如果缓冲区为空，直接加入
        if not buf:
            buf.append((rec, wave))
            continue

        # 缓冲区非空：检查温度是否变化
        prev_temp = buf[0][0].get(temp_f)

        if cur_temp == prev_temp or (cur_temp is None and prev_temp is None):
            # 温度相同，加入缓冲区
            buf.append((rec, wave))
            # 缓冲区满，合并
            if len(buf) == N:
                flush_buffer()
        else:
            # 温度变化，先处理当前缓冲区
            flush_buffer()
            # 新温度的第一个记录放入缓冲区
            buf.append((rec, wave))

    # 处理最后一批
    flush_buffer()

    # 写入输出
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"输出: {output_path} ({len(merged)} 条)")
    print(f"丢弃: {discarded} 条 (未凑齐 {N} 条同温度记录)")
    print(f"压缩比: {len(records)} → {len(merged)} ({len(merged)/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()
