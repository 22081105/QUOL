#!/usr/bin/env python3
"""
Create CSV tables for RL-GOAL results.

This script produces:
  1) Main table (matches Table 3 in the paper): mean±std for OGF, Avg L, Latency; plus success rates and stall.
  2) Appendix diagnostics table (stopping-time + reward-proxy diagnostics):
       - Tail length mean±std, TP@512 (%), TP@1024 (%)
       - EOS penalty mean±std (eos_pen) and PPL mean±std (ppl)
     Note: validity is constant 1.0 in the paper's implementation, so it is omitted.
  3) Appendix latency quantiles table (matches the style of Table 4 in the paper):
       - Median, Mean, P95, Max latency (seconds)

Chunk inputs are merged per group; missing files are skipped.

Default file patterns (15 chunks):
  Trained policy:
    policy_llama_chunk00_of_15.csv .. policy_llama_chunk14_of_15.csv
    policy_phi3_chunk00_of_15.csv  .. policy_phi3_chunk14_of_15.csv
    policy_ds_chunk00_of_15.csv    .. policy_ds_chunk14_of_15.csv

  Baseline (No Prefix):
    baseline_noprefix_llama_chunk00_of_15.csv .. baseline_noprefix_llama_chunk14_of_15.csv
    baseline_noprefix_phi3_chunk00_of_15.csv  .. baseline_noprefix_phi3_chunk14_of_15.csv
    baseline_noprefix_ds_chunk00_of_15.csv    .. baseline_noprefix_ds_chunk14_of_15.csv

Expected columns in chunk CSVs (extra columns are ignored):
  gen_len, OGF, stall, latency_sec,
  tail_len, TP@512, TP@1024,
  eos_pen, ppl

Usage:
  python3 make_rlgoal1.py
  python3 make_rlgoal1.py --dir . \
  --out_main rlgoal_results_main.csv \
  --out_appendix rlgoal_results_appendix.csv \
  --out_latency rlgoal_results_latency.csv

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

MAIN_REQUIRED = {"gen_len", "OGF", "stall", "latency_sec"}
APP_REQUIRED = {"tail_len", "TP@512", "TP@1024", "eos_pen", "ppl"}  # optional-but-used; missing => blank cells


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _pm(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def _pct(x: float, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}%"


def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _p95(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, 95)) if arr.size else 0.0


def load_chunks(base_dir: Path, template: str, n_chunks: int) -> Optional[pd.DataFrame]:
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

        # Must have at least the main required columns to be usable.
        if not MAIN_REQUIRED.issubset(df.columns):
            missing = sorted(MAIN_REQUIRED - set(df.columns))
            warn(f"{p} missing required columns {missing} (skipping file)")
            continue

        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def compute_main(df: pd.DataFrame) -> Dict[str, float]:
    ogf = pd.to_numeric(df["OGF"], errors="coerce").to_numpy(dtype=float)
    gen_len = pd.to_numeric(df["gen_len"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df["latency_sec"], errors="coerce").to_numpy(dtype=float)
    stall = pd.to_numeric(df["stall"], errors="coerce").fillna(0).to_numpy(dtype=float)
    stall = (stall != 0.0).astype(int)

    ogf_mean, ogf_std = _mean_std(ogf)
    L_mean, L_std = _mean_std(gen_len)
    lat_mean, lat_std = _mean_std(lat)

    succ1 = float((ogf >= 1).mean() * 100.0) if np.isfinite(ogf).any() else 0.0
    succ2 = float((ogf >= 2).mean() * 100.0) if np.isfinite(ogf).any() else 0.0
    succ4 = float((ogf >= 4).mean() * 100.0) if np.isfinite(ogf).any() else 0.0
    stall_rate = float((stall == 1).mean() * 100.0) if stall.size else 0.0

    return {
        "ogf_mean": ogf_mean, "ogf_std": ogf_std,
        "succ1": succ1, "succ2": succ2, "succ4": succ4,
        "stall": stall_rate,
        "L_mean": L_mean, "L_std": L_std,
        "lat_mean": lat_mean, "lat_std": lat_std,
    }


def compute_appendix_diag(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for col in ["tail_len", "eos_pen", "ppl"]:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            m, s = _mean_std(arr)
            out[f"{col}_mean"] = m
            out[f"{col}_std"] = s
        else:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan

    for col in ["TP@512", "TP@1024"]:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            arr = (arr != 0.0).astype(int)
            out[f"{col}_rate"] = float(arr.mean() * 100.0) if arr.size else 0.0
        else:
            out[f"{col}_rate"] = np.nan

    return out


def compute_latency_quantiles(df: pd.DataFrame) -> Dict[str, float]:
    lat = pd.to_numeric(df["latency_sec"], errors="coerce").to_numpy(dtype=float)
    lat = lat[~np.isnan(lat)]
    if lat.size == 0:
        return {"lat_median": 0.0, "lat_mean": 0.0, "lat_p95": 0.0, "lat_max": 0.0}
    return {
        "lat_median": float(np.median(lat)),
        "lat_mean": float(lat.mean()),
        "lat_p95": _p95(lat),
        "lat_max": float(lat.max()),
    }


def separator_row(label: str, cols: List[str]) -> Dict[str, str]:
    r = {c: "" for c in cols}
    r[cols[0]] = label
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("."), help="Directory containing chunk CSVs (default: current directory)")
    ap.add_argument("--n_chunks", type=int, default=15)

    ap.add_argument("--out_main", type=Path, default=Path("rlgoal_results_main.csv"))
    ap.add_argument("--out_appendix", type=Path, default=Path("rlgoal_results_appendix.csv"))
    ap.add_argument("--out_latency", type=Path, default=Path("rlgoal_results_latency.csv"))

    ap.add_argument("--ogf_decimals", type=int, default=2)
    ap.add_argument("--L_decimals", type=int, default=0)
    ap.add_argument("--lat_decimals", type=int, default=1)
    ap.add_argument("--pct_decimals", type=int, default=1)

    ap.add_argument("--tail_decimals", type=int, default=0)
    ap.add_argument("--ppl_decimals", type=int, default=1)
    ap.add_argument("--eos_pen_decimals", type=int, default=2)

    args = ap.parse_args()
    d = args.dir

    policy_groups = [
        ("Llama-2-7b-hf",               "policy_llama_chunk{i:02d}_of_15.csv"),
        ("Llama-2-13b-chat-hf",         "policy_llamaChat_chunk{i:02d}_of_15.csv"),
        ("Phi-3-mini-4k-instruct",      "policy_phi3_chunk{i:02d}_of_15.csv"),
        ("Deepseek-Coder-7B-Base-v1.5", "policy_ds_chunk{i:02d}_of_15.csv"),
        ("EleutherAI/pythia-6.9b",      "policy_EL6B_chunk{i:02d}_of_15.csv"),
    ]
    baseline_groups = [
        ("Llama-2-7b-hf",               "baseline_noprefix_llama_chunk{i:02d}_of_15.csv"),
        ("Llama-2-13b-chat-hf",         "baseline_noprefix_llamaChat_chunk{i:02d}_of_15.csv"),
        ("Phi-3-mini-4k-instruct",      "baseline_noprefix_phi3_chunk{i:02d}_of_15.csv"),
        ("Deepseek-Coder-7B-Base-v1.5", "baseline_noprefix_ds_chunk{i:02d}_of_15.csv"),
        ("EleutherAI/pythia-6.9b",      "baseline_noprefix_EL6B_chunk{i:02d}_of_15.csv"),
    ]
    # -------- Main table --------
    main_cols = ["LLaMA-2 Trained Policy", "Avg. OGF", "Succ.@≥1", "Succ.@≥2", "Succ.@≥4", "Stall", "Avg. L", "Latency (s)"]
    main_rows: List[Dict[str, str]] = []

    def add_main(label: str, df: pd.DataFrame) -> None:
        m = compute_main(df)
        main_rows.append({
            "LLaMA-2 Trained Policy": label,
            "Avg. OGF": _pm(m["ogf_mean"], m["ogf_std"], args.ogf_decimals),
            "Succ.@≥1": _pct(m["succ1"], args.pct_decimals),
            "Succ.@≥2": _pct(m["succ2"], args.pct_decimals),
            "Succ.@≥4": _pct(m["succ4"], args.pct_decimals),
            "Stall": _pct(m["stall"], args.pct_decimals),
            "Avg. L": _pm(m["L_mean"], m["L_std"], args.L_decimals),
            "Latency (s)": _pm(m["lat_mean"], m["lat_std"], args.lat_decimals),
        })

    for label, templ in policy_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            warn(f"No readable chunks for policy '{label}' (skipping row)")
            continue
        add_main(label, df)

    baseline_main_rows: List[Dict[str, str]] = []
    for label, templ in baseline_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            warn(f"No readable chunks for baseline '{label}' (skipping row)")
            continue
        m = compute_main(df)
        baseline_main_rows.append({
            "LLaMA-2 Trained Policy": label,
            "Avg. OGF": _pm(m["ogf_mean"], m["ogf_std"], args.ogf_decimals),
            "Succ.@≥1": _pct(m["succ1"], args.pct_decimals),
            "Succ.@≥2": _pct(m["succ2"], args.pct_decimals),
            "Succ.@≥4": _pct(m["succ4"], args.pct_decimals),
            "Stall": _pct(m["stall"], args.pct_decimals),
            "Avg. L": _pm(m["L_mean"], m["L_std"], args.L_decimals),
            "Latency (s)": _pm(m["lat_mean"], m["lat_std"], args.lat_decimals),
        })

    if baseline_main_rows:
        main_rows.append(separator_row("Baseline (No Prefix)", main_cols))
        main_rows.extend(baseline_main_rows)

    pd.DataFrame(main_rows, columns=main_cols).to_csv(args.out_main, index=False)
    print(f"Wrote main table: {args.out_main.resolve()}")

    # -------- Appendix diagnostics --------
    app_cols = [
        "Setting",
        "Avg. OGF", "Succ.@≥1", "Succ.@≥2", "Succ.@≥4", "Stall",
        "Tail len", "TP@512", "TP@1024",
        "EOS pen", "PPL",
        "Avg. L", "Latency (mean±std)",
    ]
    app_rows: List[Dict[str, str]] = []

    def add_appendix(label: str, df: pd.DataFrame) -> None:
        m = compute_main(df)
        a = compute_appendix_diag(df)
        tail = "" if np.isnan(a["tail_len_mean"]) else _pm(a["tail_len_mean"], a["tail_len_std"], args.tail_decimals)
        tp512 = "" if np.isnan(a["TP@512_rate"]) else _pct(a["TP@512_rate"], args.pct_decimals)
        tp1024 = "" if np.isnan(a["TP@1024_rate"]) else _pct(a["TP@1024_rate"], args.pct_decimals)
        eos_pen = "" if np.isnan(a["eos_pen_mean"]) else _pm(a["eos_pen_mean"], a["eos_pen_std"], args.eos_pen_decimals)
        ppl = "" if np.isnan(a["ppl_mean"]) else _pm(a["ppl_mean"], a["ppl_std"], args.ppl_decimals)

        app_rows.append({
            "Setting": label,
            "Avg. OGF": _pm(m["ogf_mean"], m["ogf_std"], args.ogf_decimals),
            "Succ.@≥1": _pct(m["succ1"], args.pct_decimals),
            "Succ.@≥2": _pct(m["succ2"], args.pct_decimals),
            "Succ.@≥4": _pct(m["succ4"], args.pct_decimals),
            "Stall": _pct(m["stall"], args.pct_decimals),
            "Tail len": tail,
            "TP@512": tp512,
            "TP@1024": tp1024,
            "EOS pen": eos_pen,
            "PPL": ppl,
            "Avg. L": _pm(m["L_mean"], m["L_std"], args.L_decimals),
            "Latency (mean±std)": _pm(m["lat_mean"], m["lat_std"], args.lat_decimals),
        })

    for label, templ in policy_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            continue
        add_appendix(label, df)

    baseline_app_rows: List[Dict[str, str]] = []
    for label, templ in baseline_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            continue
        # Build row now; insert after separator
        m = compute_main(df)
        a = compute_appendix_diag(df)
        tail = "" if np.isnan(a["tail_len_mean"]) else _pm(a["tail_len_mean"], a["tail_len_std"], args.tail_decimals)
        tp512 = "" if np.isnan(a["TP@512_rate"]) else _pct(a["TP@512_rate"], args.pct_decimals)
        tp1024 = "" if np.isnan(a["TP@1024_rate"]) else _pct(a["TP@1024_rate"], args.pct_decimals)
        eos_pen = "" if np.isnan(a["eos_pen_mean"]) else _pm(a["eos_pen_mean"], a["eos_pen_std"], args.eos_pen_decimals)
        ppl = "" if np.isnan(a["ppl_mean"]) else _pm(a["ppl_mean"], a["ppl_std"], args.ppl_decimals)

        baseline_app_rows.append({
            "Setting": label,
            "Avg. OGF": _pm(m["ogf_mean"], m["ogf_std"], args.ogf_decimals),
            "Succ.@≥1": _pct(m["succ1"], args.pct_decimals),
            "Succ.@≥2": _pct(m["succ2"], args.pct_decimals),
            "Succ.@≥4": _pct(m["succ4"], args.pct_decimals),
            "Stall": _pct(m["stall"], args.pct_decimals),
            "Tail len": tail,
            "TP@512": tp512,
            "TP@1024": tp1024,
            "EOS pen": eos_pen,
            "PPL": ppl,
            "Avg. L": _pm(m["L_mean"], m["L_std"], args.L_decimals),
            "Latency (mean±std)": _pm(m["lat_mean"], m["lat_std"], args.lat_decimals),
        })

    if baseline_app_rows:
        app_rows.append({c: "" for c in app_cols} | {"Setting": "Baseline (No Prefix)"})
        app_rows.extend(baseline_app_rows)

    pd.DataFrame(app_rows, columns=app_cols).to_csv(args.out_appendix, index=False)
    print(f"Wrote appendix diagnostics: {args.out_appendix.resolve()}")

    # -------- Appendix latency quantiles --------
    lat_cols = ["Setting", "Median lat. (s)", "Mean lat. (s)", "P95 lat. (s)", "Max lat. (s)"]
    lat_rows: List[Dict[str, str]] = []

    def add_latency(label: str, df: pd.DataFrame) -> None:
        q = compute_latency_quantiles(df)
        lat_rows.append({
            "Setting": label,
            "Median lat. (s)": f"{q['lat_median']:.2f}",
            "Mean lat. (s)": f"{q['lat_mean']:.2f}",
            "P95 lat. (s)": f"{q['lat_p95']:.2f}",
            "Max lat. (s)": f"{q['lat_max']:.2f}",
        })

    for label, templ in policy_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            continue
        add_latency(label, df)

    baseline_lat_rows: List[Dict[str, str]] = []
    for label, templ in baseline_groups:
        df = load_chunks(d, templ, args.n_chunks)
        if df is None:
            continue
        q = compute_latency_quantiles(df)
        baseline_lat_rows.append({
            "Setting": label,
            "Median lat. (s)": f"{q['lat_median']:.2f}",
            "Mean lat. (s)": f"{q['lat_mean']:.2f}",
            "P95 lat. (s)": f"{q['lat_p95']:.2f}",
            "Max lat. (s)": f"{q['lat_max']:.2f}",
        })

    if baseline_lat_rows:
        lat_rows.append({"Setting": "Baseline (No Prefix)", "Median lat. (s)": "", "Mean lat. (s)": "", "P95 lat. (s)": "", "Max lat. (s)": ""})
        lat_rows.extend(baseline_lat_rows)

    pd.DataFrame(lat_rows, columns=lat_cols).to_csv(args.out_latency, index=False)
    print(f"Wrote appendix latency quantiles: {args.out_latency.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
