"""
Balance dataset by voltage interval with sliding window augmentation.

Reads a JSONL file, groups records by voltage interval (e.g. -40~-20, -20~0,
0~20, ...), and uses sliding window to balance each interval toward a target
count N.

Usage:
    uv run python scripts/data/balance_data.py
    uv run python scripts/data/balance_data.py --input data/exported_data.jsonl --output data/5000.jsonl --n 5000
    uv run python scripts/data/balance_data.py --help
"""

import sys
import json
import math
import os
import random
from collections import defaultdict

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_voltage(v) -> float | None:
    """Parse ACTUAL_VOLTAGE field to float."""
    if v is None:
        return None
    s = str(v).strip().lower().replace("v", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def bucket_label(v: float, interval: int) -> str:
    """Get interval label for a voltage, e.g. -40~-20."""
    low = math.floor(v / interval) * interval
    return f"{low}~{low + interval}"


def generate_windows(wave: list[float], window: int, stride: int, max_count: int) -> list[list[float]]:
    """Generate sliding windows from a waveform."""
    starts = list(range(0, window, stride))[:max_count]
    return [wave[s:s + window] for s in starts]


def build_record(rec: dict, wave_window: list[float]) -> dict:
    """Build a new record with windowed waveform."""
    out = dict(rec)
    out["RTU_REGS_P00_WAVE_DATA"] = ",".join(str(x) for x in wave_window)
    return out


@click.command(help="Balance dataset by voltage interval with sliding window augmentation.")
@click.option("--input", "input_path", default=None,
              help="Input JSONL file path (default: data/exported_data.jsonl)")
@click.option("--output", "output_path", default=None,
              help="Output JSONL file path (default: data/5000.jsonl)")
@click.option("--n", "N", type=int, default=None,
              help="Target number of samples per voltage interval (default: 5000)")
@click.option("--interval", "interval_w", type=int, default=None,
              help="Voltage interval width, e.g. 20 means -40~-20, -20~0 (default: 20)")
@click.option("--window", type=int, default=None,
              help="Sliding window size (default: 256)")
@click.option("--stride", "min_stride", type=int, default=None,
              help="Minimum sliding window stride (default: 10)")
@click.option("--seed", type=int, default=None,
              help="Random seed (default: 42)")
def main(input_path, output_path, N, interval_w, window, min_stride, seed):
    """Balance dataset by voltage interval with sliding window augmentation."""
    # Interactive prompts for missing values
    if input_path is None:
        input_path = click.prompt("Input JSONL file", default="data/exported_data.jsonl")
    if output_path is None:
        output_path = click.prompt("Output JSONL file", default="data/5000.jsonl")
    if N is None:
        N = click.prompt("Target samples per interval", type=int, default=5000)
    if interval_w is None:
        interval_w = click.prompt("Voltage interval width", type=int, default=20)
    if window is None:
        window = click.prompt("Sliding window size", type=int, default=256)
    if min_stride is None:
        min_stride = click.prompt("Minimum stride", type=int, default=10)
    if seed is None:
        seed = click.prompt("Random seed", type=int, default=42)

    random.seed(seed)
    max_slides = window // min_stride

    # Load data
    from src.swa.signal_process.loader import load_jsonl
    records = load_jsonl(input_path)
    click.echo(f"Loaded {len(records)} records from {input_path}\n")

    # Group by voltage interval
    buckets = defaultdict(list)
    skipped = 0
    for rec in records:
        v = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
        if v is None:
            skipped += 1
            continue
        buckets[bucket_label(v, interval_w)].append(rec)

    click.echo(f"Grouped into {len(buckets)} voltage intervals ({skipped} skipped)\n")

    # Shuffle each bucket in-place with fixed seed → deterministic order
    for recs in buckets.values():
        random.shuffle(recs)

    # Process each interval
    output_records = []
    bucket_stats = []

    for label in sorted(buckets.keys(), key=_sort_key):
        recs = buckets[label]
        base_count = len(recs)
        max_total = base_count * max_slides

        if base_count >= N:
            # Enough raw records, take first N (already shuffled)
            for rec in recs[:N]:
                wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
                wave = [float(x) for x in wave_str.split(",")][:512]
                out = build_record(rec, wave[:window])
                output_records.append(out)
            bucket_stats.append((label, base_count, N, min_stride, 1))
            continue

        # Need augmentation — calculate stride
        if max_total <= N:
            # Even max augmentation can't reach N, use max
            actual_stride = min_stride
        else:
            # Calculate stride to hit ~N
            target_windows = (N + base_count - 1) // base_count  # ceil
            actual_stride = max(min_stride, window // target_windows)

        actual_max_slides = min(max_slides, window // actual_stride)

        generated = []
        for rec in recs:
            wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
            wave = [float(x) for x in wave_str.split(",")][:512]
            windows_list = generate_windows(wave, window, actual_stride, actual_max_slides)
            for w in windows_list:
                out = build_record(rec, w)
                generated.append(out)

        # Trim if over target (take first N, already deterministic)
        if len(generated) > N:
            generated = generated[:N]

        output_records.extend(generated)
        bucket_stats.append((label, base_count, len(generated), actual_stride, actual_max_slides))

    # Print summary
    click.echo(f"{'Interval':>12}  {'Raw':>6}  {'Final':>8}  {'Stride':>7}  {'Win/Rec':>7}")
    click.echo("-" * 50)
    total_raw = 0
    total_final = 0
    for label, raw, final, s, wpr in bucket_stats:
        click.echo(f"{label:>12}  {raw:>6}  {final:>8}  {s:>7}  {wpr:>7}")
        total_raw += raw
        total_final += final
    click.echo("-" * 50)
    click.echo(f"{'Total':>12}  {total_raw:>6}  {total_final:>8}")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    click.echo(f"\nSaved {len(output_records)} records to {output_path}")


def _sort_key(label: str) -> int:
    """Sort intervals by their lower bound."""
    low = label.split("~")[0]
    try:
        return int(low)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
