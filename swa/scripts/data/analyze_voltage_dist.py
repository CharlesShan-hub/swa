#!/usr/bin/env python
"""Analyze voltage distribution: how many samples per interval."""

import sys
import os
import math
from collections import Counter

print("SCRIPT LOADED:", __file__, file=sys.stderr)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

import click


def parse_voltage(v) -> float | None:
    """Parse ACTUAL_VOLTAGE field to float."""
    if v is None:
        return None

    s = str(v).strip().lower()
    s = (
        s.replace("v", "")
         .replace("mv", "")
         .replace(" ", "")
    )

    if s in ("", "n/a", "null", "--", "none"):
        return None

    try:
        return float(s)
    except ValueError:
        return None


@click.command(help="Analyze voltage distribution by interval.")
@click.option(
    "--dest",
    required=True,
    help="Input JSONL file path",
)
@click.option(
    "--interval",
    type=int,
    default=20,
    help="Voltage interval width",
)
@click.option(
    "--sort-by",
    type=click.Choice(["count", "voltage"]),
    default="count",
    help="Sort order: count or voltage",
)
def main(dest: str, interval: int, sort_by: str):
    """Analyze voltage distribution by interval."""

    if not os.path.exists(dest):
        click.secho(f"❌ Input file not found: {dest}", fg="red")
        sys.exit(1)

    from src.swa.signal_process.loader import load_jsonl

    all_rec = load_jsonl(dest)
    click.echo(f"Loaded {len(all_rec)} records from {dest}")

    cnt = Counter()
    skipped = 0

    for r in all_rec:
        v = parse_voltage(r.get("ACTUAL_VOLTAGE"))
        if v is None:
            skipped += 1
            continue

        bucket_low = math.floor(v / interval) * interval
        label = f"{bucket_low}~{bucket_low + interval}"
        cnt[label] += 1

    if not cnt:
        click.secho("❌ No valid voltage data found.", fg="red")
        sys.exit(1)

    total = sum(cnt.values())

    def sort_key(item):
        label, n = item
        if sort_by == "voltage":
            return int(label.split("~")[0])
        return -n

    click.echo(f"{'Interval':>12}  {'Count':>6}  {'Pct':>6}  {'Note'}")
    click.echo("-" * 55)

    for label, n in sorted(cnt.items(), key=sort_key):
        pct = n / total * 100
        note = ""
        if n < 100:
            note = "<-- rare"
        elif n < 500:
            note = "<-- low sample"
        click.echo(f"{label:>12}  {n:>6}  {pct:>5.1f}%  {note}")

    click.echo()
    if skipped:
        click.secho(f"Skipped {skipped} unparseable records", fg="yellow")


if __name__ == "__main__":
    main()