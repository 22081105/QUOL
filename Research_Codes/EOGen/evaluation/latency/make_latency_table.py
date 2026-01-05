#!/usr/bin/env python3
"""
make_latency_table_v2.py

Latency summary tables (median/mean/p95/max) for:
  - EOGen
  - EOGen-suffix
  - Baselines: Repeat-style / Inf. babble / Random short / WizardLM

Key detail (baselines):
Your baseline CSVs have `prompt_key` but no family column. In run_baselines.py, keys are:
  prompt_key = md5(f"{family}:{tag}")
This script reproduces that exact keyset logic to label baseline rows.

You MUST pass the same counts used in run_baselines.py:
  --num_random_prompts (default 100)
  --num_wizard_prompts (default 100)

Paths:
  phi3:
    test_eogen: eogen_results_phi3_chunk00_of_15.csv .. chunk14_of_15.csv
    suffix:     phi_suffix_results_chunk00_of_15.csv  .. chunk14_of_15.csv
    baseline:   baseline_results_phi3_chunk00_of_15.csv .. chunk14_of_15.csv
  ds:
    same structure with 'ds'
  llama:
    eogen/suffix use 'llama' but baseline filenames use 'llama2'

Run:
  python3 make_latency_table_v2.py --out_dir /srv/scratch/$USER/latency/lat_tables
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def expand_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def get_latency_series(df: pd.DataFrame, path_hint: str) -> pd.Series | None:
    lat_col = find_col(
        df,
        ["latency_sec", "latency_s", "latency", "latency (s)", "Latency (s)", "latency_seconds"],
    )
    if lat_col is None:
        warn(f"{path_hint}: missing latency column. Columns={list(df.columns)}")
        return None
    s = pd.to_numeric(df[lat_col], errors="coerce").dropna()
    if s.empty:
        warn(f"{path_hint}: latency column '{lat_col}' empty after coercion")
        return None
    return s


def load_chunks(base_dir: Path, template: str, n_chunks: int) -> pd.DataFrame | None:
    dfs: list[pd.DataFrame] = []
    for i in range(n_chunks):
        p = base_dir / template.format(i=i, n=n_chunks)
        if not p.exists():
            warn(f"Missing file: {p} (skipping)")
            continue
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            warn(f"Failed to read {p}: {e} (skipping)")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def latency_stats(lat: np.ndarray) -> dict[str, float]:
    lat = lat.astype(float)
    return {
        "median": float(np.median(lat)),
        "mean": float(np.mean(lat)),
        "p95": float(np.quantile(lat, 0.95)),
        "max": float(np.max(lat)),
    }


def make_row(label: str, lat: pd.Series, decimals: int) -> dict[str, str]:
    st = latency_stats(lat.to_numpy(dtype=float))
    return {
        "Prompt source": label,
        "Median lat. (s)": fmt(st["median"], decimals),
        "Mean lat. (s)": fmt(st["mean"], decimals),
        "P95 lat. (s)": fmt(st["p95"], decimals),
        "Max lat. (s)": fmt(st["max"], decimals),
    }


# -------------------------
# Baseline keysets (must match run_baselines.py)
# -------------------------
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _repeat_style_tags() -> List[str]:
    tags: List[str] = []
    tags += [f"repeat_v{i}" for i in range(1, 6)]
    tags += [f"recursion_v{i}" for i in range(1, 6)]
    tags += [f"count_v{i}" for i in range(1, 6)]
    tags += [f"longtext_v{i}" for i in range(1, 6)]
    tags += [f"code_v{i}" for i in range(1, 6)]
    return tags


def _infinite_babble_tags() -> List[str]:
    return [f"infinite_babble_v{i:02d}" for i in range(1, 31)]


def _random_short_tags(n: int) -> List[str]:
    return [f"random_{i:03d}" for i in range(n)]


def _wizard_tags(n: int) -> List[str]:
    return [f"wizard_{i:05d}" for i in range(n)]


def _keys(family: str, tags: Iterable[str]) -> Set[str]:
    return {_md5(f"{family}:{tag}") for tag in tags}


def baseline_keysets(num_random_prompts: int, num_wizard_prompts: int) -> Dict[str, Set[str]]:
    return {
        "Repeat-style": _keys("repeat_style", _repeat_style_tags()),
        "Inf. babble": _keys("infinite_babble", _infinite_babble_tags()),
        "Random short": _keys("random_short", _random_short_tags(num_random_prompts)),
        "WizardLM": _keys("wizard_instruct", _wizard_tags(num_wizard_prompts)),
    }


def label_baseline_rows(df: pd.DataFrame, model_name: str, n_random: int, n_wizard: int) -> pd.DataFrame:
    if "prompt_key" not in df.columns:
        warn(f"{model_name}/Baselines: missing prompt_key; baselines will be unlabeled")
        out = df.copy()
        out["_fam"] = "Baselines (unlabeled)"
        return out

    ks = baseline_keysets(n_random, n_wizard)

    def assign(k: str) -> str:
        k = str(k).strip()
        for fam, s in ks.items():
            if k in s:
                return fam
        return "Baselines (unlabeled)"

    out = df.copy()
    out["_fam"] = out["prompt_key"].map(assign)
    return out


def summarize_model(
    model_name: str,
    root: Path,
    tag: str,
    n_chunks: int,
    decimals: int,
    n_random: int,
    n_wizard: int,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    # EOGen
    eogen_dir = root / "test_eogen"
    eogen_df = load_chunks(eogen_dir, f"eogen_results_{tag}_chunk{{i:02d}}_of_{{n}}.csv", n_chunks)
    if eogen_df is not None:
        lat = get_latency_series(eogen_df, f"{model_name}/EOGen")
        if lat is not None:
            rows.append(make_row("EOGen", lat, decimals))
    else:
        warn(f"{model_name}: no EOGen chunks under {eogen_dir}")

    # EOGen-suffix
    suffix_dir = root / "suffix"
    suffix_tpl = "phi_suffix_results_chunk{i:02d}_of_{n}.csv" if model_name == "phi3" else f"{tag}_suffix_results_chunk{{i:02d}}_of_{{n}}.csv"
    suffix_df = load_chunks(suffix_dir, suffix_tpl, n_chunks)
    if suffix_df is not None:
        lat = get_latency_series(suffix_df, f"{model_name}/EOGen-suffix")
        if lat is not None:
            rows.append(make_row("EOGen-suffix", lat, decimals))
    else:
        warn(f"{model_name}: no suffix chunks under {suffix_dir}")

    # Baselines
    baseline_dir = root / "baseline"
    baseline_tag = "llama2" if model_name == "llama" else tag
    base_df = load_chunks(baseline_dir, f"baseline_results_{baseline_tag}_chunk{{i:02d}}_of_{{n}}.csv", n_chunks)

    if base_df is None:
        warn(f"{model_name}: no baseline chunks under {baseline_dir}")
    else:
        if "row_type" in base_df.columns:
            base_df = base_df[base_df["row_type"].astype(str).str.upper().eq("TRIAL")].copy()

        base_df = label_baseline_rows(base_df, model_name, n_random, n_wizard)

        order = ["Repeat-style", "Inf. babble", "Random short", "WizardLM"]
        for fam in order:
            sub = base_df[base_df["_fam"] == fam]
            if sub.empty:
                continue
            lat = get_latency_series(sub, f"{model_name}/Baseline:{fam}")
            if lat is not None:
                rows.append(make_row(fam, lat, decimals))

        unl = base_df[base_df["_fam"] == "Baselines (unlabeled)"]
        if not unl.empty:
            warn(f"{model_name}: {len(unl)} baseline rows did not match keysets. Check n_random/n_wizard.")
            lat = get_latency_series(unl, f"{model_name}/Baselines (unlabeled)")
            if lat is not None:
                rows.append(make_row("Baselines (unlabeled)", lat, decimals))

    return pd.DataFrame(rows, columns=["Prompt source", "Median lat. (s)", "Mean lat. (s)", "P95 lat. (s)", "Max lat. (s)"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".", help="Where to write output CSV tables")
    ap.add_argument("--phi_root", type=str, default="/srv/scratch/$USER/phi_test")
    ap.add_argument("--ds_root", type=str, default="/srv/scratch/$USER/ds_test")
    ap.add_argument("--llama_root", type=str, default="/srv/scratch/$USER/llama_test")
    ap.add_argument("--models", nargs="+", default=["phi3", "ds", "llama"], choices=["phi3", "ds", "llama"])
    ap.add_argument("--n_chunks", type=int, default=15)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--num_random_prompts", type=int, default=100)
    ap.add_argument("--num_wizard_prompts", type=int, default=100)
    args = ap.parse_args()

    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "phi3": (expand_path(args.phi_root), "phi3"),
        "ds": (expand_path(args.ds_root), "ds"),
        "llama": (expand_path(args.llama_root), "llama"),  # baseline uses llama2 (handled)
    }

    for m in args.models:
        root, tag = spec[m]
        df = summarize_model(
            model_name=m,
            root=root,
            tag=tag,
            n_chunks=args.n_chunks,
            decimals=args.decimals,
            n_random=args.num_random_prompts,
            n_wizard=args.num_wizard_prompts,
        )
        out_csv = out_dir / f"latency_table_{m}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n=== {m} ===")
        print(df.to_string(index=False))
        print(f"Wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
