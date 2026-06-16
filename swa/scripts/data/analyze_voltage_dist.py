"""Analyze voltage distribution: how many samples per interval."""
import sys, os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from collections import Counter

import click
from src.swa.signal_process.loader import load_jsonl


def parse_voltage(v) -> float | None:
    """Parse ACTUAL_VOLTAGE field to float."""
    if v is None:
        return None
    s = str(v).strip().lower().replace("v", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


@click.command(help="Analyze voltage distribution by interval.")
@click.option("--input", "input_path", default="data/exported_data.jsonl",
              help="Input JSONL file path")
@click.option("--interval", "interval_w", type=int, default=20,
              help="Voltage interval width")
def main(input_path, interval_w):
    """Analyze voltage distribution by interval."""
    all_rec = load_jsonl(input_path)
    print(f"Loaded {len(all_rec)} records from {input_path}\n")

    cnt = Counter()
    skipped = 0
    for r in all_rec:
        v = parse_voltage(r.get("ACTUAL_VOLTAGE"))
        if v is None:
            skipped += 1
            continue
        bucket_low = math.floor(v / interval_w) * interval_w
        label = f"{bucket_low}~{bucket_low + interval_w}"
        cnt[label] += 1

    total = sum(cnt.values())
    print(f"{'Interval':>12}  {'Count':>6}  {'Pct':>6}  {'Note'}")
    print("-" * 55)
    for label, n in sorted(cnt.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        note = ""
        if n < 100:
            note = "<-- rare"
        elif n < 500:
            note = "<-- low sample"
        print(f"{label:>12}  {n:>6}  {pct:>5.1f}%  {note}")

    print()
    if skipped:
        print(f"(Skipped {skipped} unparseable records)")


if __name__ == "__main__":
    main()
