#!/usr/bin/env python3
"""
Compare memorization metrics between two output directories (e.g. baseline vs unlearned).
Recursively finds all *_metrics.json and *_cross_seed.json files regardless of nesting.

Usage:
    python compare_metrics.py output/cap3d output/nemo
    python compare_metrics.py output/cap3d output/nemo --csv results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def find_json_files(root: Path, suffix: str):
    return list(root.rglob(f"*{suffix}"))


def extract_scalars(d: dict, prefix="") -> dict:
    """Recursively extract scalar (float/int) leaf values from a nested dict."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(extract_scalars(v, key))
        # skip lists (trajectories, visualizations)
    return out


def load_metrics(root: Path):
    """
    Returns two dicts:
      per_seed: {metric_key: [values]}
      cross_seed: {metric_key: [values]}
    """
    per_seed = defaultdict(list)
    cross_seed = defaultdict(list)

    for f in find_json_files(root, "_metrics.json"):
        try:
            data = json.loads(f.read_text())
            metrics = data.get("metrics", data)  # handle both wrapped and flat
            scalars = extract_scalars(metrics)
            for k, v in scalars.items():
                # skip non-metric keys
                if any(x in k for x in ["prompt_idx", "num_views", "image_resolution"]):
                    continue
                per_seed[k].append(v)
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)

    for f in find_json_files(root, "_cross_seed.json"):
        try:
            data = json.loads(f.read_text())
            scalars = extract_scalars(data)
            for k, v in scalars.items():
                if k in ("memorized",):
                    continue
                cross_seed[k].append(v)
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)

    return per_seed, cross_seed


def summarize(values: list) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def print_comparison(dir_a: Path, dir_b: Path, csv_path=None):
    print(f"Loading {dir_a} ...")
    ps_a, cs_a = load_metrics(dir_a)
    print(f"  per-seed files: {sum(len(v) for v in ps_a.values())//max(len(ps_a),1)} metrics × {len(next(iter(ps_a.values()), []))} samples")

    print(f"Loading {dir_b} ...")
    ps_b, cs_b = load_metrics(dir_b)

    all_keys = sorted(set(ps_a) | set(ps_b))
    all_cs_keys = sorted(set(cs_a) | set(cs_b))

    col = 42
    w = 14

    def header(title):
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        print(f"{'Metric':<{col}} {'A mean':>{w}} {'A std':>{w}} {'B mean':>{w}} {'B std':>{w}} {'Δ%':>{w}}")
        print(f"{'-'*col} {'-'*w} {'-'*w} {'-'*w} {'-'*w} {'-'*w}")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    def delta(a, b):
        if a is None or b is None or a == 0:
            return "—"
        return f"{(b-a)/abs(a)*100:+.1f}%"

    rows = []

    header(f"Per-seed metrics  |  A={dir_a.name}  B={dir_b.name}")
    for k in all_keys:
        sa = summarize(ps_a.get(k, []))
        sb = summarize(ps_b.get(k, []))
        # shorten key for display
        short = k.replace("metrics.", "").replace("Hessian_SAIL_Metric.", "Hessian.")
        short = short.replace("Noise_Difference_Norm.", "NoiseDiff.")
        short = short[:col-1]
        d = delta(sa["mean"], sb["mean"])
        print(f"{short:<{col}} {fmt(sa['mean']):>{w}} {fmt(sa['std']):>{w}} {fmt(sb['mean']):>{w}} {fmt(sb['std']):>{w}} {d:>{w}}")
        rows.append(("per_seed", k, sa["n"], sa["mean"], sa["std"], sb["n"], sb["mean"], sb["std"]))

    header(f"Cross-seed metrics  |  A={dir_a.name}  B={dir_b.name}")
    for k in all_cs_keys:
        sa = summarize(cs_a.get(k, []))
        sb = summarize(cs_b.get(k, []))
        short = k[:col-1]
        d = delta(sa["mean"], sb["mean"])
        print(f"{short:<{col}} {fmt(sa['mean']):>{w}} {fmt(sa['std']):>{w}} {fmt(sb['mean']):>{w}} {fmt(sb['std']):>{w}} {d:>{w}}")
        rows.append(("cross_seed", k, sa["n"], sa["mean"], sa["std"], sb["n"], sb["mean"], sb["std"]))

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["type", "metric", "A_n", "A_mean", "A_std", "B_n", "B_mean", "B_std", "delta_pct"])
            for r in rows:
                typ, k, an, am, as_, bn, bm, bs = r
                d = f"{(bm-am)/abs(am)*100:+.1f}" if am and am != 0 and bm is not None else ""
                w_.writerow([typ, k, an, am, as_, bn, bm, bs, d])
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", help="Baseline output directory")
    ap.add_argument("dir_b", help="Unlearned output directory")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()
    print_comparison(Path(args.dir_a), Path(args.dir_b), args.csv)
