#!/usr/bin/env python3
"""
Change default_root accordingly
Typical usage (on the cluster):
  python3 make_table.py --num_chunks 15 --out_csv llama_overgen_table.csv
"""

import argparse
import hashlib
import os
from pathlib import Path
import glob
from typing import Dict, Iterable, List, Set, Tuple, Optional, Sequence
import numpy as np
import pandas as pd


# -------------------------
# helpers
# -------------------------

def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _sample_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def _pm(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def _pct(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}%"


def _expand_csv_args(items: Optional[Sequence[str]]) -> List[Path]:
    """
    Expand CSV args that may include:
      - a literal path
      - a glob pattern (contains * ? [)
      - a comma-separated list of paths/globs
    """
    if not items:
        return []

    out: List[Path] = []
    for it in items:
        if it is None:
            continue
        parts = [p.strip() for p in str(it).split(",") if p.strip()]
        for p in parts:
            if any(ch in p for ch in ["*", "?", "["]):
                out.extend([Path(x).resolve() for x in sorted(glob.glob(p))])
            else:
                out.append(Path(p).resolve())

    # de-dup while preserving order
    seen = set()
    dedup: List[Path] = []
    for p in out:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def _discover_inputs(dir_path: Path, prefix: str, num_chunks: Optional[int]) -> List[Path]:
    """
    Priority:
      1) If num_chunks provided:   <prefix>_chunk*_of_{num_chunks:02d}.csv
      2) Any chunked pattern:      <prefix>_chunk*_of_*.csv
      3) Merged single file:      <prefix>.csv
      4) Merged alt filename:     <prefix>_merged.csv
      5) If exactly one match of  <prefix>*.csv, use it
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {dir_path}")

    if num_chunks is not None:
        pat = f"{prefix}_chunk*_of_{num_chunks:02d}.csv"
        files = sorted(dir_path.glob(pat))
        if files:
            return files

    # fallback: any chunk count
    files = sorted(dir_path.glob(f"{prefix}_chunk*_of_*.csv"))
    if files:
        return files

    merged1 = dir_path / f"{prefix}.csv"
    if merged1.exists():
        return [merged1]

    merged2 = dir_path / f"{prefix}_merged.csv"
    if merged2.exists():
        return [merged2]

    matches = sorted(dir_path.glob(f"{prefix}*.csv"))
    if len(matches) == 1:
        return [matches[0]]

    raise FileNotFoundError(
        f"Could not find inputs for prefix '{prefix}' under {dir_path}.\n"
        f"Tried chunked patterns '{prefix}_chunk*_of_XX.csv' and merged '{prefix}.csv' / '{prefix}_merged.csv'."
    )



def _load_trials(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load one or more CSVs and return a minimal dataframe with columns:
      - prompt_key (string identifier for the prompt)
      - OGF (float)

    Supports different schemas:
      - If 'prompt_key' exists: use it.
      - Else if 'prompt' exists: hash prompt text -> prompt_key.
      - Else if 'instruction_index' exists: use it as a stable id.
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        # Filter TRIAL rows if row_type exists
        if "row_type" in df.columns:
            m = df["row_type"].astype(str).str.upper() == "TRIAL"
            df = df[m].copy()
        else:
            df = df.copy()

        if "OGF" not in df.columns:
            raise ValueError(f"{p} must contain column 'OGF'. Found: {list(df.columns)}")

        # Ensure we have a prompt identifier
        if "prompt_key" in df.columns:
            df["prompt_key"] = df["prompt_key"].astype(str)
        elif "prompt" in df.columns:
            df["prompt_key"] = df["prompt"].astype(str).map(_md5)
        elif "instruction_index" in df.columns:
            df["prompt_key"] = df["instruction_index"].astype(str)
        else:
            raise ValueError(
                f"{p} must contain a prompt id column: 'prompt_key' OR 'prompt' OR 'instruction_index'. "
                f"Found: {list(df.columns)}"
            )

        df["OGF"] = pd.to_numeric(df["OGF"], errors="coerce")
        df = df.dropna(subset=["OGF", "prompt_key"])

        dfs.append(df[["prompt_key", "OGF"]].copy())

    if not dfs:
        return pd.DataFrame(columns=["prompt_key", "OGF"])

    return pd.concat(dfs, ignore_index=True)



def _agg(trials: pd.DataFrame) -> Dict[str, object]:
    ogf = trials["OGF"].to_numpy(dtype=float)
    out: Dict[str, object] = {
        "#Prompts": int(trials["prompt_key"].nunique()),
        "ogf_mean": float(np.mean(ogf)) if ogf.size else 0.0,
        "ogf_std": _sample_std(ogf),
    }
    for k in (1, 2, 3, 4):
        out[f"S@{k}"] = float(np.mean(ogf >= k) * 100.0) if ogf.size else 0.0
    return out

# -------------------------
# main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="If set, expect chunk files with suffix _of_YY. If omitted, auto-detect chunked or merged CSVs.",
    )

    ap.add_argument("--ogf_decimals", type=int, default=2, help="Decimals for mean±std formatting.")
    ap.add_argument("--pct_decimals", type=int, default=1, help="Decimals for success percentages.")

    # Input roots
    ap.add_argument("--eogen_dir", type=Path, default=None)
    ap.add_argument("--suffix_dir", type=Path, default=None)
    
    ap.add_argument(
        "--eogen_csv",
        action="append",
        default=None,
        help="Explicit EOGen CSV path(s) or glob(s). Overrides directory/prefix discovery.",
    )
    ap.add_argument(
        "--suffix_csv",
        action="append",
        default=None,
        help="Explicit suffix CSV path(s) or glob(s). Overrides directory/prefix discovery.",
    )

    # File prefixes (basename before _chunkXX_of_YY.csv)
    ap.add_argument("--eogen_prefix", type=str, default="eogen_results_llama")
    ap.add_argument("--suffix_prefix", type=str, default="llama_suffix_results")



    # Output
    ap.add_argument("--out_csv", type=Path, default=Path("llama_overgen_table.csv"))

    args = ap.parse_args()

    default_root = Path(__file__).resolve().parent
    eogen_dir = args.eogen_dir or (default_root / "test_eogen")
    suffix_dir = args.suffix_dir or (default_root / "suffix")

    # Load chunked trials
    eogen_explicit = _expand_csv_args(args.eogen_csv)
    suffix_explicit = _expand_csv_args(args.suffix_csv)
    
    eogen_files = eogen_explicit or _discover_inputs(eogen_dir, args.eogen_prefix, args.num_chunks)
    suffix_files = suffix_explicit or _discover_inputs(suffix_dir, args.suffix_prefix, args.num_chunks)
    
    eogen_trials = _load_trials(eogen_files)
    suffix_trials = _load_trials(suffix_files)

    # Build table rows
    rows: List[Dict[str, object]] = []

    def add_row(label: str, trials: pd.DataFrame):
        m = _agg(trials)
        rows.append({
            "Prompt source": label,
            "#Prompts": m["#Prompts"],
            "Avg. OGF": _pm(m["ogf_mean"], m["ogf_std"], args.ogf_decimals),
            "Succ. @≥1": _pct(round(float(m["S@1"]), args.pct_decimals), args.pct_decimals),
            "Succ. @≥2": _pct(round(float(m["S@2"]), args.pct_decimals), args.pct_decimals),
            "Succ. @≥3": _pct(round(float(m["S@3"]), args.pct_decimals), args.pct_decimals),
            "Succ. @≥4": _pct(round(float(m["S@4"]), args.pct_decimals), args.pct_decimals),
        })

    # EOGen and suffix
    add_row("EOGen", eogen_trials)
    add_row("EOGen-suffix", suffix_trials)


    out_df = pd.DataFrame(rows, columns=[
        "Prompt source", "#Prompts", "Avg. OGF",
        "Succ. @≥1", "Succ. @≥2", "Succ. @≥3", "Succ. @≥4",
    ])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
