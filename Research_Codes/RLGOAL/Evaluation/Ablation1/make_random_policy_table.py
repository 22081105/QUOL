#!/usr/bin/env python3
"""
Make a 2-column CSV table (like the screenshot) by merging 15 chunk files:
  random_policy_chunk00_of_15.csv ... random_policy_chunk14_of_15.csv

Runs from the *current directory* by default (no hardcoded paths).

Output columns:
  Metric, <label>

Metrics computed follow the same aggregation logic used in your make_rlgoal.py:
- Avg. OGF: mean ± std (ddof=1)
- S@≥1/2/4 (%): fraction(OGF >= threshold) * 100
- Stall (%): fraction(stall != 0) * 100
- Avg. L: mean ± std
- Latency (s): mean ± std
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


ALIASES = {
    "gen_len": ["gen_len", "victim_gen_len", "length", "L"],
    "latency_sec": ["latency_sec", "latency", "latency_s", "lat_s"],
    "OGF": ["OGF", "ogf"],
    "stall": ["stall", "is_stall", "stalled"],
}


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def pick_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for c in ALIASES.get(key, [key]):
        if c in df.columns:
            return c
    return None


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def fmt_pm(mean: float, std: float, decimals: int) -> str:
    # screenshot style uses spaces around ±
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def load_chunks(base_dir: Path, template: str, n_chunks: int) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for i in range(n_chunks):
        p = base_dir / template.format(i=i)
        if not p.exists():
            warn(f"Missing file: {p} (skipping)")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            warn(f"Failed to read {p}: {e} (skipping)")
            continue
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No readable chunk files found for template='{template}' in dir='{base_dir}'."
        )
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    c_ogf = pick_col(df, "OGF")
    c_len = pick_col(df, "gen_len")
    c_stall = pick_col(df, "stall")
    c_lat = pick_col(df, "latency_sec")

    missing = [
        k
        for k, c in [("OGF", c_ogf), ("gen_len", c_len), ("stall", c_stall), ("latency_sec", c_lat)]
        if c is None
    ]
    if missing:
        raise KeyError(f"Missing required columns (or aliases): {missing}. Available columns: {list(df.columns)}")

    ogf = pd.to_numeric(df[c_ogf], errors="coerce").to_numpy(dtype=float)
    gen_len = pd.to_numeric(df[c_len], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df[c_lat], errors="coerce").to_numpy(dtype=float)

    stall_raw = pd.to_numeric(df[c_stall], errors="coerce").fillna(0).to_numpy(dtype=float)
    stall = (stall_raw != 0.0).astype(int)

    ogf_mean, ogf_std = mean_std(ogf)
    L_mean, L_std = mean_std(gen_len)
    lat_mean, lat_std = mean_std(lat)

    finite_any = np.isfinite(ogf).any()
    succ1 = float((ogf >= 1).mean() * 100.0) if finite_any else 0.0
    succ2 = float((ogf >= 2).mean() * 100.0) if finite_any else 0.0
    succ4 = float((ogf >= 4).mean() * 100.0) if finite_any else 0.0
    stall_rate = float(stall.mean() * 100.0) if stall.size else 0.0

    return {
        "ogf_mean": ogf_mean,
        "ogf_std": ogf_std,
        "succ1": succ1,
        "succ2": succ2,
        "succ4": succ4,
        "stall": stall_rate,
        "L_mean": L_mean,
        "L_std": L_std,
        "lat_mean": lat_mean,
        "lat_std": lat_std,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("."), help="Directory containing chunk CSVs (default: .)")
    ap.add_argument("--n_chunks", type=int, default=15, help="Total chunks (default: 15 for 00..14)")
    ap.add_argument(
        "--template",
        type=str,
        default="random_policy_chunk{i:02d}_of_15.csv",
        help="Filename template (default: random_policy_chunk{i:02d}_of_15.csv)",
    )
    ap.add_argument(
        "--label",
        type=str,
        default="Uniform-random policy (LLaMA-2)",
        help="2nd column header label",
    )
    ap.add_argument("--out", type=Path, default=Path("random_policy_table.csv"), help="Output CSV filename")

    args = ap.parse_args()

    df = load_chunks(args.dir, args.template, args.n_chunks)
    m = compute_metrics(df)

    rows = [
        ("Avg. OGF", fmt_pm(m["ogf_mean"], m["ogf_std"], decimals=2)),
        ("S@≥1 (%)", f"{m['succ1']:.1f}"),
        ("S@≥2 (%)", f"{m['succ2']:.1f}"),
        ("S@≥4 (%)", f"{m['succ4']:.1f}"),
        ("Stall (%)", f"{m['stall']:.1f}"),
        ("Avg. $L$", fmt_pm(m["L_mean"], m["L_std"], decimals=0)),
        ("Latency (s)", fmt_pm(m["lat_mean"], m["lat_std"], decimals=1)),
    ]

    out_df = pd.DataFrame(rows, columns=["Metric", args.label])
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out.resolve()}  (rows={len(out_df)}, merged_records={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
