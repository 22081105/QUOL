#!/usr/bin/env python3
"""
plot_rlgoal_figures.py

Camera-ready visualizations for RL-GOAL (method + results). This version mirrors the
*exact* group/file patterns used by make_rlgoal.py.

Victim-model groups (from make_rlgoal.py):
  - Llama-2-7b-hf                 policy_llama_chunk00_of_15.csv .. policy_llama_chunk14_of_15.csv
  - Llama-2-13b-chat-hf           policy_llamaChat_chunk00_of_15.csv .. policy_llamaChat_chunk14_of_15.csv
  - Phi-3-mini-4k-instruct        policy_phi3_chunk00_of_15.csv .. policy_phi3_chunk14_of_15.csv
  - Deepseek-Coder-7B-Base-v1.5   policy_ds_chunk00_of_15.csv .. policy_ds_chunk14_of_15.csv
  - EleutherAI/pythia-6.9b        policy_EL6B_chunk00_of_15.csv .. policy_EL6B_chunk14_of_15.csv

And matching no-prefix baselines:
  - baseline_noprefix_llama_chunk*.csv / baseline_noprefix_llamaChat_chunk*.csv / ...

If some groups are missing in DIR, they will be skipped (with warnings).

Outputs (PDF + PNG by default):
  1) rlgoal_method_schematic.{pdf,png}
  2) rlgoal_results_summary.{pdf,png}        (OGF, S@>=2, Stall%, Mean latency)
  3) rlgoal_latency_box.{pdf,png}            (latency distributions, log y-axis)
  4) rlgoal_ogf_latency_tradeoff.{pdf,png}   (per-model scatter: OGF vs latency)

Usage:
  python3 plot_rlgoal_figures.py --dir /srv/scratch/$USER/rlgoal_chunks --out_dir /srv/scratch/$USER/plots/rlgoal

Optional:
  python3 plot_rlgoal_figures.py --include "Phi-3-mini-4k-instruct" "Llama-2-7b-hf"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# -------------------------
# Console helpers
# -------------------------
def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def expand_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


# -------------------------
# Data loading
# -------------------------
REQUIRED_COLS = {"OGF", "gen_len", "stall", "latency_sec"}


def load_chunks(base_dir: Path, template: str, n_chunks: int) -> Optional[pd.DataFrame]:
    """
    Load chunk CSVs from base_dir using template. Template must contain {i:02d} and {n}.
    Drops non-TRIAL rows if row_type exists.
    """
    dfs: List[pd.DataFrame] = []
    for i in range(n_chunks):
        p = base_dir / template.format(i=i, n=n_chunks)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            warn(f"Failed to read {p}: {e} (skipping)")
            continue

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            warn(f"{p}: missing required columns {sorted(missing)} (skipping)")
            continue

        if "row_type" in df.columns:
            df = df[df["row_type"].astype(str).str.upper().eq("TRIAL")].copy()

        dfs.append(df)

    if not dfs:
        return None

    out = pd.concat(dfs, ignore_index=True)

    # Coerce key columns
    for c in ["OGF", "gen_len", "stall", "latency_sec"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["OGF", "latency_sec"], how="any")

    # Stall: treat nonzero as stall
    out["stall"] = (out["stall"].fillna(0).astype(float) != 0.0).astype(int)

    return out


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    ogf = df["OGF"].to_numpy(dtype=float)
    lat = df["latency_sec"].to_numpy(dtype=float)
    stall = df["stall"].to_numpy(dtype=int)

    out: Dict[str, float] = {}
    out["ogf_mean"] = float(np.mean(ogf)) if ogf.size else 0.0
    out["succ2"] = float((ogf >= 2).mean() * 100.0) if ogf.size else 0.0
    out["stall"] = float((stall == 1).mean() * 100.0) if stall.size else 0.0
    out["lat_mean"] = float(np.mean(lat)) if lat.size else 0.0
    out["lat_median"] = float(np.median(lat)) if lat.size else 0.0
    out["lat_p95"] = float(np.percentile(lat, 95)) if lat.size else 0.0
    return out


# -------------------------
# Plot styling
# -------------------------
def set_rc():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, make_png: bool = True, make_pdf: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_pdf:
        fig.savefig(out_dir / f"{stem}.pdf")
    if make_png:
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


# -------------------------
# Figure 1: Method schematic
# -------------------------
def fig_method_schematic(out_dir: Path, make_png: bool, make_pdf: bool) -> None:
    set_rc()
    fig = plt.figure(figsize=(8.6, 4.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Panel A: Training loop
    ax.text(0.05, 0.92, "A. RL-GOAL training loop", fontweight="bold")

    boxes = [
        (0.05, 0.70, 0.23, 0.14, "Sample goal\n(target length / OGF)"),
        (0.31, 0.70, 0.26, 0.14, "Policy proposes\nprefix tokens"),
        (0.60, 0.70, 0.30, 0.14, "Victim generate\n(prefix → completion)"),
        (0.05, 0.50, 0.85, 0.14, "Compute reward\n(length/OGF + EOS penalty + diagnostics)"),
        (0.05, 0.30, 0.85, 0.14, "PPO update\n(policy + value) with replay/HER"),
    ]

    for (x, y, w, h, t) in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                facecolor="white",
            )
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center")

    # arrows
    arrow_kw = dict(arrowstyle="->", lw=1.0, color="black")
    ax.annotate("", xy=(0.31, 0.77), xytext=(0.28, 0.77), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.60, 0.77), xytext=(0.57, 0.77), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.47, 0.64), xytext=(0.47, 0.70), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.47, 0.44), xytext=(0.47, 0.50), arrowprops=arrow_kw)

    ax.annotate("", xy=(0.12, 0.70), xytext=(0.12, 0.44), arrowprops=dict(arrowstyle="<-", lw=1.0))

    # Panel B: Goal-conditioned attacker
    ax.text(0.05, 0.20, "B. Goal-conditioned attacker (compact policy/value)", fontweight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.05),
            0.85,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            facecolor="white",
        )
    )
    ax.text(
        0.475,
        0.11,
        "Inputs: goal g + state summary → Policy πφ(a|s,g) (prefix tokens) and Value Vψ(s,g)\n"
        "Victim model is query-only; only compact πφ/Vψ are trained.",
        ha="center",
        va="center",
    )

    save_fig(fig, out_dir, "rlgoal_method_schematic", make_png, make_pdf)


# -------------------------
# Figure 2: Results summary (bar small-multiples)
# -------------------------
def short_label(label: str) -> str:
    s = label
    s = s.replace("Llama-2-", "LLaMA-2-")
    s = s.replace("-hf", "")
    s = s.replace("Phi-3-mini-4k-instruct", "Phi-3-mini")
    s = s.replace("Deepseek-Coder-7B-Base-v1.5", "DeepSeek-Coder")
    s = s.replace("EleutherAI/pythia-6.9b", "Pythia-6.9B")
    s = s.replace("13b-chat", "13B-chat")
    s = s.replace("7b", "7B")
    return s


def fig_results_summary(
    out_dir: Path,
    groups: List[Tuple[str, Dict[str, str]]],
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    make_png: bool,
    make_pdf: bool,
) -> None:
    set_rc()

    labels = [g[0] for g in groups]
    x = np.arange(len(labels))
    width = 0.37

    # Compute metrics for each (model, setting)
    metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    for model_label, templs in groups:
        for setting in ["RL-GOAL", "No-prefix"]:
            df = dfs.get((model_label, setting))
            if df is None or df.empty:
                continue
            metrics[(model_label, setting)] = compute_metrics(df)

    # Keep only models with at least one setting present
    present = []
    for model_label in labels:
        if (model_label, "RL-GOAL") in metrics or (model_label, "No-prefix") in metrics:
            present.append(model_label)

    if not present:
        warn("No readable RL-GOAL/no-prefix data found for any group; skipping results figures.")
        return

    labels = present
    groups = [(lab, dict()) for lab in labels]  # just for label order
    x = np.arange(len(labels))

    def arr(key: str, setting: str) -> np.ndarray:
        out = []
        for lab in labels:
            out.append(metrics.get((lab, setting), {}).get(key, np.nan))
        return np.array(out, dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 5.8))
    axs = axs.ravel()

    panels = [
        ("Average OGF", "ogf_mean"),
        ("Success @ OGF≥2 (%)", "succ2"),
        ("Stall rate (%)", "stall"),
        ("Mean latency (s)", "lat_mean"),
    ]

    for ax, (title, key) in zip(axs, panels):
        a_rl = arr(key, "RL-GOAL")
        a_np = arr(key, "No-prefix")

        ax.bar(x - width / 2, a_rl, width, label="RL-GOAL")
        ax.bar(x + width / 2, a_np, width, label="No-prefix")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(s) for s in labels], rotation=25, ha="right")
        ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)

    axs[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_fig(fig, out_dir, "rlgoal_results_summary", make_png, make_pdf)


# -------------------------
# Figure 3: Latency distributions (boxplots)
# -------------------------
def fig_latency_box(
    out_dir: Path,
    labels: List[str],
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    make_png: bool,
    make_pdf: bool,
) -> None:
    set_rc()

    # Build boxplot data in alternating RL-GOAL / No-prefix order per model
    data = []
    ticklabels = []
    for lab in labels:
        for setting in ["RL-GOAL", "No-prefix"]:
            df = dfs.get((lab, setting))
            if df is None or df.empty:
                data.append(np.array([np.nan]))
            else:
                data.append(df["latency_sec"].to_numpy(dtype=float))
            ticklabels.append(f"{short_label(lab)}\n{setting}")

    # If everything is empty, skip
    if all(np.isnan(np.nanmedian(d)) for d in data):
        warn("No latency data to plot (boxplots).")
        return

    fig = plt.figure(figsize=(10.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.boxplot(
        data,
        showfliers=False,
        whis=(5, 95),
    )
    ax.set_yscale("log")
    ax.set_ylabel("Latency (s) [log scale]")
    ax.set_xticks(np.arange(1, len(ticklabels) + 1))
    ax.set_xticklabels(ticklabels, rotation=25, ha="right")
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)

    fig.tight_layout()
    save_fig(fig, out_dir, "rlgoal_latency_box", make_png, make_pdf)


# -------------------------
# Figure 4: OGF–latency tradeoff (scatter)
# -------------------------
def fig_ogf_latency_tradeoff(
    out_dir: Path,
    labels: List[str],
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    make_png: bool,
    make_pdf: bool,
) -> None:
    set_rc()

    n = len(labels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(10.2, 3.2 * nrows))
    for idx, lab in enumerate(labels):
        ax = fig.add_subplot(nrows, ncols, idx + 1)

        for setting in ["RL-GOAL", "No-prefix"]:
            df = dfs.get((lab, setting))
            if df is None or df.empty:
                continue
            # light subsample to keep PDFs small
            sub = df.sample(n=min(2000, len(df)), random_state=0) if len(df) > 2000 else df
            ax.scatter(sub["OGF"].to_numpy(dtype=float), sub["latency_sec"].to_numpy(dtype=float), s=6, alpha=0.35, label=setting)

        ax.set_yscale("log")
        ax.set_xlabel("OGF")
        ax.set_ylabel("Latency (s) [log]")
        ax.set_title(short_label(lab))
        ax.grid(True, linewidth=0.6, alpha=0.30)
        ax.legend(frameon=True)

    fig.tight_layout()
    save_fig(fig, out_dir, "rlgoal_ogf_latency_tradeoff", make_png, make_pdf)


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directory containing RL-GOAL chunk CSVs")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory for figures")
    ap.add_argument("--n_chunks", type=int, default=15, help="Number of chunks per group (default: 15)")
    ap.add_argument("--no_png", action="store_true", help="Disable PNG outputs")
    ap.add_argument("--no_pdf", action="store_true", help="Disable PDF outputs")
    ap.add_argument("--include", nargs="*", default=None, help="Optional subset of group labels to plot (exact match)")

    args = ap.parse_args()
    base_dir = expand_path(args.dir)
    out_dir = expand_path(args.out_dir)

    make_png = not args.no_png
    make_pdf = not args.no_pdf

    # Mirrors make_rlgoal.py group patterns (policy_groups / baseline_groups)
    # See make_rlgoal.py lines defining policy_groups and baseline_groups.
    GROUPS: List[Tuple[str, Dict[str, str]]] = [
        (
            "Llama-2-7b-hf",
            {
                "RL-GOAL": "policy_llama_chunk{i:02d}_of_{n}.csv",
                "No-prefix": "baseline_noprefix_llama_chunk{i:02d}_of_{n}.csv",
            },
        ),
        (
            "Llama-2-13b-chat-hf",
            {
                "RL-GOAL": "policy_llamaChat_chunk{i:02d}_of_{n}.csv",
                "No-prefix": "baseline_noprefix_llamaChat_chunk{i:02d}_of_{n}.csv",
            },
        ),
        (
            "Phi-3-mini-4k-instruct",
            {
                "RL-GOAL": "policy_phi3_chunk{i:02d}_of_{n}.csv",
                "No-prefix": "baseline_noprefix_phi3_chunk{i:02d}_of_{n}.csv",
            },
        ),
        (
            "Deepseek-Coder-7B-Base-v1.5",
            {
                "RL-GOAL": "policy_ds_chunk{i:02d}_of_{n}.csv",
                "No-prefix": "baseline_noprefix_ds_chunk{i:02d}_of_{n}.csv",
            },
        ),
        (
            "EleutherAI/pythia-6.9b",
            {
                "RL-GOAL": "policy_EL6B_chunk{i:02d}_of_{n}.csv",
                "No-prefix": "baseline_noprefix_EL6B_chunk{i:02d}_of_{n}.csv",
            },
        ),
    ]

    if args.include:
        wanted = set(args.include)
        GROUPS = [g for g in GROUPS if g[0] in wanted]
        if not GROUPS:
            warn("--include matched no group labels; proceeding with empty group list (method schematic only).")

    # Always produce the method schematic
    fig_method_schematic(out_dir, make_png, make_pdf)

    # Load data frames
    dfs: Dict[Tuple[str, str], pd.DataFrame] = {}
    present_labels: List[str] = []
    for model_label, templs in GROUPS:
        any_loaded = False
        for setting, templ in templs.items():
            df = load_chunks(base_dir, templ, args.n_chunks)
            if df is None or df.empty:
                continue
            dfs[(model_label, setting)] = df
            any_loaded = True
        if any_loaded:
            present_labels.append(model_label)
        else:
            warn(f"No readable chunks for group '{model_label}' under {base_dir} (skipping in result plots)")

    if not present_labels:
        warn("No RL-GOAL/no-prefix groups were found; wrote only rlgoal_method_schematic.*")
        return 0

    fig_results_summary(out_dir, [(lab, {}) for lab in present_labels], dfs, make_png, make_pdf)
    fig_latency_box(out_dir, present_labels, dfs, make_png, make_pdf)
    fig_ogf_latency_tradeoff(out_dir, present_labels, dfs, make_png, make_pdf)

    print(f"Wrote figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
