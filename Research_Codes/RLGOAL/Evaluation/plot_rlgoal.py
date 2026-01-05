#!/usr/bin/env python3
"""
 python plot_rlgoal1.py   --in_dir   /evaluation   --out_dir  /plots   --make_rlgoal make_rlgoal.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator


# --- Color palette (Tol Bright) ---
# Reference palette (7 colors): 4477AA, EE6677, 228833, CCBB44, 66CCEE, AA3377, BBBBBB
# We only need two contrasting, colorblind-safe colors for the two conditions.
TOL_BRIGHT = {
    "blue":  "#4477AA",
    "red":   "#EE6677",
    "green": "#228833",
    "yellow":"#CCBB44",
    "cyan":  "#66CCEE",
    "purple":"#AA3377",
    "grey":  "#BBBBBB",
}

COND_COLORS = {
    "RL-GOAL": TOL_BRIGHT["blue"],
    "No-prefix": TOL_BRIGHT["red"],
}

PREFERRED_ORDER = [
    "Llama-2-7b-hf",
    "Llama-2-13b-chat-hf",
    "Phi-3-mini-4k-instruct",
    "EleutherAI/pythia-6.9b",
    "Falcon-7B",
    "DeepSeek-Coder",
    "DeepSeek",
    "DS",
]

TAG_TO_DISPLAY = {
    # --- Llama-2 7B base ---
    "llama": "Llama-2-7b-hf",
    "llama2": "Llama-2-7b-hf",
    "llama2_7b": "Llama-2-7b-hf",
    "llama-2-7b": "Llama-2-7b-hf",
    "llama7b": "Llama-2-7b-hf",
    "llama_7b": "Llama-2-7b-hf",
    "llama-2-7b-hf": "Llama-2-7b-hf",

    # --- Llama-2 13B chat ---
    # policy_llamaChat_* -> tag "llamaChat" -> _clean_tag => "llamachat"
    "llamachat": "Llama-2-13b-chat-hf",
    "llama_chat": "Llama-2-13b-chat-hf",
    "llama13b": "Llama-2-13b-chat-hf",
    "llama_13b": "Llama-2-13b-chat-hf",
    "llama13bchat": "Llama-2-13b-chat-hf",
    "llama_13b_chat": "Llama-2-13b-chat-hf",
    "llama2_13b_chat": "Llama-2-13b-chat-hf",
    "llama-2-13b-chat": "Llama-2-13b-chat-hf",
    "llama2-13b-chat": "Llama-2-13b-chat-hf",
    "llama-2-13b-chat-hf": "Llama-2-13b-chat-hf",

    # --- Phi-3 mini 4k instruct ---
    "phi3": "Phi-3-mini-4k-instruct",
    "phi3mini": "Phi-3-mini-4k-instruct",
    "phi3_mini": "Phi-3-mini-4k-instruct",
    "phi-3-mini": "Phi-3-mini-4k-instruct",
    "phi-3-mini-4k": "Phi-3-mini-4k-instruct",
    "phi3-mini-4k-instruct": "Phi-3-mini-4k-instruct",

    # --- EleutherAI Pythia 6.9B ---
    # policy_EL6B_* -> tag "EL6B" -> _clean_tag => "el6b"
    "el6b": "EleutherAI/pythia-6.9b",
    "pythia": "EleutherAI/pythia-6.9b",
    "pythia6.9b": "EleutherAI/pythia-6.9b",
    "pythia_6.9b": "EleutherAI/pythia-6.9b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "eleutherai_pythia_6_9b": "EleutherAI/pythia-6.9b",

    # --- DeepSeek ---
    "ds": "DeepSeek-Coder",
    "deepseek": "DeepSeek-Coder",
    "deepseekcoder": "DeepSeek-Coder",
    "deepseek-coder": "DeepSeek-Coder",
}



@dataclass(frozen=True)
class GroupSpec:
    prefix: str          # file prefix (e.g., policy_llama)
    model_tag: str       # short tag (e.g., llama)
    model_name: str      # display name (e.g., LLaMA-2-7B)
    condition: str       # RL-GOAL or No-prefix
    label: str           # human-readable label


def _clean_tag(tag: str) -> str:
    tag = tag.strip().lower()
    tag = re.sub(r"[^a-z0-9]+", "_", tag)
    tag = tag.strip("_")
    return tag


def tag_to_display(tag: str) -> str:
    t = _clean_tag(tag)
    return TAG_TO_DISPLAY.get(t, tag)


def _log_seconds_formatter() -> FuncFormatter:
    def _fmt(y, _pos):
        # show clean seconds (avoid 10^x)
        if y == 0:
            return "0"
        # Prefer integers when close
        if y >= 1 and abs(y - round(y)) / max(1.0, y) < 1e-9:
            return str(int(round(y)))
        # Otherwise compact
        return f"{y:g}"
    return FuncFormatter(_fmt)


def set_log_seconds_axis(ax: plt.Axes, label: str = "Latency (s)") -> None:
    """
    Log-scale latency axis, but tick labels are plain seconds (1, 10, 100, ...),
    i.e., no 10^k / scientific notation. Keeps minor ticks unlabeled.
    """
    ax.set_yscale("log")

    # Major ticks at powers of 10; minor ticks at 2,3,5,7 × 10^k (unlabeled)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 5, 7)))

    def _fmt(y: float, _pos: int) -> str:
        if not np.isfinite(y) or y <= 0:
            return ""
        if y >= 1000:
            return f"{y:,.0f}"
        if abs(y - round(y)) < 1e-9:
            return f"{int(round(y))}"
        return f"{y:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.set_ylabel(label)
    try:
        ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass


def plot_latency_distribution(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: Path,
    max_jitter: int = 1200,
):
    """
    Variant B: ONE combined plot (all models) with box + jitter.
    Y-axis is log-scaled for heavy tails, but tick labels are rendered as seconds (1,10,100,...),
    avoiding 10^k notation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    out_dir.mkdir(parents=True, exist_ok=True)

    # Colorblind-safe Tol Bright subset (hand-picked)
    # No-prefix: warm (salmon), RL-GOAL: cool (blue)
    COLORS = {
        "No-prefix": "#EE6677",
        "RL-GOAL":   "#4477AA",
    }

    # Flatten into a consistent order
    models = sort_models(list(all_data.keys()))
    conds = ["No-prefix", "RL-GOAL"]

    # Build data in plotting order: for each model -> [No-prefix, RL-GOAL]
    groups = []
    labels = []
    for m in models:
        for c in conds:
            df = all_data[m].get(c, None)
            if df is None or df.empty or "latency_sec" not in df.columns:
                y = np.array([], dtype=float)
            else:
                y = pd.to_numeric(df["latency_sec"], errors="coerce").dropna().to_numpy()
            groups.append(y)
            labels.append((m, c))

    # Figure
    fig, ax = plt.subplots(figsize=(10.8, 4.6))

    # X positions: grouped by model, two conditions per model
    x = []
    xlabels = []
    for i, m in enumerate(models):
        # two slots per model
        x0 = i * 3.0
        x.extend([x0, x0 + 1.0])
        xlabels.extend(["No-prefix", "RL-GOAL"])

    # Boxplot
    bp = ax.boxplot(
        groups,
        positions=x,
        widths=0.85,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(edgecolor="black", linewidth=1.0),
    )

    # Color the boxes by condition
    for i, box in enumerate(bp["boxes"]):
        _, c = labels[i]
        box.set_facecolor(COLORS.get(c, "#BBBBBB"))
        box.set_alpha(0.22)

    # Jittered points (cap count per group via max_jitter)
    rng = np.random.default_rng(0)
    for i, y in enumerate(groups):
        if y.size == 0:
            continue
        if y.size > max_jitter:
            idx = rng.choice(y.size, size=max_jitter, replace=False)
            yy = y[idx]
        else:
            yy = y

        xx = x[i] + rng.uniform(-0.18, 0.18, size=yy.size)
        _, c = labels[i]
        ax.scatter(
            xx, yy,
            s=10,
            alpha=0.16,
            linewidths=0,
            color=COLORS.get(c, "#777777"),
            zorder=2,
        )

    # Log scale but format as plain seconds
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

    # Cosmetic grid (behind)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.08)
    ax.set_axisbelow(True)

    # X ticks: model names centered under pairs
    model_centers = [i * 3.0 + 0.5 for i in range(len(models))]
    ax.set_xticks(model_centers)
    ax.set_xticklabels(models, rotation=20, ha="right")

    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency distributions")

    # Legend (single, clean)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=COLORS["No-prefix"], markeredgecolor="black", alpha=0.35,
                   label="No-prefix"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=COLORS["RL-GOAL"], markeredgecolor="black", alpha=0.35,
                   label="RL-GOAL"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14)
    fig.savefig(out_dir / "dist_B.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "dist_B.pdf", bbox_inches="tight")

    plt.close(fig)

def plot_tradeoff(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: Path,
    *,
    min_n: int = 30,
    max_points_per_violin: int = 400,  # jitter overlay cap (keeps file size sane)
):
    """
    Tradeoff alternative: Violin plot of latency amplification.

    For each model:
      - Compute baseline = median latency under No-prefix
      - Plot amplification = latency / baseline for both conditions
      - Two violins per model (No-prefix vs RL-GOAL), log-y
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- appealing, paper-friendly pair (teal + lavender) ---
    COLORS = {
        "No-prefix": "#4DB6AC",  # soft teal
        "RL-GOAL":   "#8E7CC3",  # light purple / lavender
    }

    # panel order (uses your current model keys in all_data)
    preferred = ["LLaMA-2-7B", "LLaMA-2-13B-chat", "Phi-3-mini", "EleutherAI/pythia-6.9b"]
    models = [m for m in preferred if m in all_data] + [m for m in all_data.keys() if m not in preferred]
    models = models[:4]

    # force Pythia to the rightmost position
    for k in ("Pythia-6.9B", "EleutherAI/pythia-6.9b"):
        if k in models:
            models = [m for m in models if m != k] + [k]
            break

    def _lat(df: pd.DataFrame) -> np.ndarray:
        if df is None or df.empty or "latency_sec" not in df.columns:
            return np.array([], dtype=float)
        v = pd.to_numeric(df["latency_sec"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        v = v[v > 0]
        return v.to_numpy(dtype=float)

    rng = np.random.default_rng(0)

    # Build per-(model,cond) amplification arrays
    amp = {}  # (model, cond) -> np.ndarray
    for m in models:
        base_lat = _lat(all_data[m].get("No-prefix", None))
        if base_lat.size < min_n:
            continue
        base_med = float(np.median(base_lat))
        if not np.isfinite(base_med) or base_med <= 0:
            continue

        for cond in ("No-prefix", "RL-GOAL"):
            vv = _lat(all_data[m].get(cond, None))
            if vv.size < min_n:
                amp[(m, cond)] = np.array([], dtype=float)
            else:
                a = vv / base_med
                a = a[np.isfinite(a) & (a > 0)]
                amp[(m, cond)] = a

    # ---- Plot ----
    fig = plt.figure(figsize=(7.6, 3.6))
    ax = plt.gca()

    x0 = np.arange(len(models))
    off = 0.18
    width = 0.30

    # Violin data in plotting order
    positions = []
    data = []
    colors = []
    labels = []
    for i, m in enumerate(models):
        for cond in ("No-prefix", "RL-GOAL"):
            vals = amp.get((m, cond), np.array([], dtype=float))
            if vals.size == 0:
                # still reserve the position so spacing stays consistent
                vals = np.array([np.nan], dtype=float)
            p = x0[i] + (-off if cond == "No-prefix" else off)
            positions.append(p)
            data.append(vals)
            colors.append(COLORS[cond])
            labels.append((m, cond))

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=width,
        showmeans=False,
        showextrema=False,
        showmedians=True,
    )

    # style bodies + median lines
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor("none")
        body.set_alpha(0.28)

    vp["cmedians"].set_linewidth(2.2)
    vp["cmedians"].set_alpha(0.95)

    # optional: light jitter overlay to show density (subsampled)
    for p, vals, col in zip(positions, data, colors):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > max_points_per_violin:
            idx = rng.choice(vals.size, size=max_points_per_violin, replace=False)
            vals = vals[idx]
        xx = p + rng.uniform(-0.06, 0.06, size=vals.size)
        ax.scatter(xx, vals, s=7, alpha=0.08, linewidths=0, color=col, zorder=2)

    # axes/labels
    ax.set_xticks(x0)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Latency amplification (× No-prefix median)")
    ax.set_title("Latency amplification distribution")

    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.set_axisbelow(True)

    # legend
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["No-prefix"], lw=6, label="No-prefix"),
            Line2D([0], [0], color=COLORS["RL-GOAL"],   lw=6, label="RL-GOAL"),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    fig.subplots_adjust(top=0.88, bottom=0.28, left=0.10, right=0.98)

    fig.savefig(out_dir / "rlgoal_tradeoff_A.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "rlgoal_tradeoff_A.pdf", bbox_inches="tight")
    plt.close(fig)



def plot_overview_metrics(all_data: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path):
    """
    Overview metrics (style = attached figure):
      - 1x4 panels
      - grouped bars (No-prefix vs RL-GOAL)
      - single shared legend centered on top (no duplicates)
      - clean titles, no overlap
      - mean latency uses log-y, but ticks show plain seconds (not 10^k)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    out_dir.mkdir(parents=True, exist_ok=True)

    # Match attached look: pale yellow vs purple
    COLORS = {
        "No-prefix": "#D8D27A",  # pale yellow
        "RL-GOAL":   "#6B5BA6",  # purple
    }
    conds = ["No-prefix", "RL-GOAL"]
    models = sort_models(list(all_data.keys()))


    # --- helpers ---
    def _numcol_mean(df: pd.DataFrame, col: str) -> float:
        if df is None or df.empty or col not in df.columns:
            return np.nan
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(v.mean()) if len(v) else np.nan

    def _rate01(df: pd.DataFrame, col: str) -> float:
        """Mean of a 0/1 indicator -> fraction in [0,1]."""
        if df is None or df.empty or col not in df.columns:
            return np.nan
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(v.mean()) if len(v) else np.nan

    # --- compute metrics per (model, condition) ---
    avg_ogf = {}
    succ_ge2_pct = {}
    stall_proxy_frac = {}
    mean_lat = {}

    for m in models:
        for c in conds:
            df = all_data[m].get(c, None)

            avg_ogf[(m, c)] = _numcol_mean(df, "OGF")

            # Success@OGF>=2 (%)
            if df is not None and not df.empty and "OGF" in df.columns:
                ogf = pd.to_numeric(df["OGF"], errors="coerce")
                succ_ge2_pct[(m, c)] = float((ogf >= 2.0).mean() * 100.0)
            else:
                succ_ge2_pct[(m, c)] = np.nan

            # Stall proxy: fraction in [0,1] (matches your attached y-scale 0..0.5)
            if df is not None and not df.empty:
                if "stall" in df.columns:
                    stall_proxy_frac[(m, c)] = _rate01(df, "stall")
                elif ("cap_hit" in df.columns) and ("saw_eos" in df.columns):
                    cap = pd.to_numeric(df["cap_hit"], errors="coerce").fillna(0) > 0
                    eos = pd.to_numeric(df["saw_eos"], errors="coerce").fillna(0) > 0
                    stall_proxy_frac[(m, c)] = float((cap & (~eos)).mean())
                else:
                    stall_proxy_frac[(m, c)] = np.nan
            else:
                stall_proxy_frac[(m, c)] = np.nan

            mean_lat[(m, c)] = _numcol_mean(df, "latency_sec")

    # --- plotting ---
    x = np.arange(len(models))
    w = 0.36  # bar width

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), gridspec_kw={"wspace": 0.35})

    def grouped_bars(ax, title, yvals_map, ylog=False, ylim=None):
        y0 = [yvals_map.get((m, "No-prefix"), np.nan) for m in models]
        y1 = [yvals_map.get((m, "RL-GOAL"), np.nan) for m in models]

        ax.bar(x - w/2, y0, width=w, color=COLORS["No-prefix"], label="No-prefix")
        ax.bar(x + w/2, y1, width=w, color=COLORS["RL-GOAL"], label="RL-GOAL")

        ax.set_title(title, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        if ylog:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

        if ylim is not None:
            ax.set_ylim(*ylim)

    grouped_bars(axes[0], "Avg. OGF", avg_ogf, ylog=False)
    grouped_bars(axes[1], r"S@OGF$\geq$2 (%)", succ_ge2_pct, ylog=False, ylim=(0.0, 100.0))
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])

    grouped_bars(axes[2], "Stall rate (fraction)", stall_proxy_frac, ylog=False, ylim=(0.0, 0.5))
    axes[2].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    fig.subplots_adjust(top=0.78, bottom=0.30, left=0.06, right=0.995, wspace=0.35)

    grouped_bars(axes[3], "Mean latency (s)", mean_lat, ylog=True)

    # Single shared legend (top center) — no duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))

    # leave room for the figure-level legend above
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.06, right=0.99, wspace=0.35)
    fig.savefig(out_dir / "overview_D.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "overview_D.pdf", bbox_inches="tight")

    plt.close(fig)


def style_axes(ax):
    ax.grid(True, axis="y", which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, axis="y", which="minor", alpha=0.12, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_groups_from_make_rlgoal(make_rlgoal_path: Path) -> List[GroupSpec]:
    """
    Attempts to parse a `groups = [...]` list in make_rlgoal.py with 4-tuples:
      ("Label", "model_tag", "file_prefix", "policy_key")
    """
    text = make_rlgoal_path.read_text(encoding="utf-8", errors="ignore")

    # Find tuple entries inside the groups list
    # Example: ("RL-GOAL (LLaMA-2)", "llama", "policy_llama", "policy_llama")
    tup_re = re.compile(
        r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
    )
    specs: List[GroupSpec] = []
    for m in tup_re.finditer(text):
        label, model_tag, file_prefix, _policy_key = m.groups()
        cond = "RL-GOAL" if file_prefix.startswith("policy_") else "No-prefix" if file_prefix.startswith("baseline_noprefix_") else "Unknown"
        model_name = tag_to_display(model_tag)
        specs.append(GroupSpec(prefix=file_prefix, model_tag=model_tag, model_name=model_name, condition=cond, label=label))

    return specs


def scan_groups_from_dir(in_dir: Path) -> List[GroupSpec]:
    """
    Scans for chunk/merged files to infer prefixes and model tags.

    Recognized naming:
      policy_<tag>_chunk*.csv or policy_<tag>_merged.csv
      baseline_noprefix_<tag>_chunk*.csv or baseline_noprefix_<tag>_merged.csv
    """
    files = list(in_dir.glob("*.csv"))
    prefixes = set()
    for f in files:
        name = f.name
        # Prefer merged or chunk naming
        name = re.sub(r"_chunk\d+_of_\d+\.csv$", ".csv", name)
        name = re.sub(r"_chunk\d+_of_\d+\.CSV$", ".csv", name)
        name = re.sub(r"_merged\.csv$", ".csv", name)
        if name.startswith("policy_") or name.startswith("baseline_noprefix_"):
            prefixes.add(name[:-4])  # strip .csv

    specs: List[GroupSpec] = []
    for pref in sorted(prefixes):
        if pref.startswith("policy_"):
            tag = pref[len("policy_") :]
            cond = "RL-GOAL"
        elif pref.startswith("baseline_noprefix_"):
            tag = pref[len("baseline_noprefix_") :]
            cond = "No-prefix"
        else:
            continue
        model_name = tag_to_display(tag)
        label = f"{cond} ({model_name})"
        specs.append(GroupSpec(prefix=pref, model_tag=tag, model_name=model_name, condition=cond, label=label))
    return specs


def resolve_group_files(in_dir: Path, prefix: str) -> List[Path]:
    """
    Prefer merged file if present; otherwise collect chunks.
    """
    merged = in_dir / f"{prefix}_merged.csv"
    if merged.exists():
        return [merged]

    # Accept a wide variety of chunk naming
    chunk_globs = [
        f"{prefix}_chunk*_of_*.csv",
        f"{prefix}_chunk*.csv",
        f"{prefix}*chunk*_of_*.csv",
    ]
    out: List[Path] = []
    for g in chunk_globs:
        out.extend(sorted(in_dir.glob(g)))
    # De-dup while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def load_trials(csv_paths: List[Path]) -> pd.DataFrame:
    if not csv_paths:
        return pd.DataFrame()

    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Filter to trial rows
    if "row_type" in df.columns:
        df = df[df["row_type"].astype(str).str.upper() == "TRIAL"].copy()

    # Normalize/ensure required columns
    # latency column
    if "latency_sec" not in df.columns:
        for alt in ["latency", "latency_s", "latency_seconds", "Latency (s)"]:
            if alt in df.columns:
                df["latency_sec"] = df[alt]
                break

    # OGF column
    if "OGF" not in df.columns:
        for alt in ["ogf", "max_ogf"]:
            if alt in df.columns:
                df["OGF"] = df[alt]
                break

    # stall column
    if "stall" not in df.columns:
        for alt in ["stall_flag", "is_stall", "Stall"]:
            if alt in df.columns:
                df["stall"] = df[alt]
                break

    # Coerce numeric
    for c in ["latency_sec", "OGF", "stall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[c for c in ["latency_sec", "OGF"] if c in df.columns])

    # Stall: normalize to {0,1}
    if "stall" in df.columns:
        df["stall"] = (df["stall"].fillna(0).astype(float) != 0.0).astype(int)

    return df


def sort_models(model_names: List[str]) -> List[str]:
    def key(m):
        try:
            return (0, PREFERRED_ORDER.index(m))
        except ValueError:
            return (1, m.lower())
    return sorted(model_names, key=key)


def binned_tradeoff(df: pd.DataFrame, ogf_step: float = 0.25, min_n: int = 30) -> pd.DataFrame:
    """
    Returns binned median/IQR latency vs OGF.
    """
    if df.empty:
        return pd.DataFrame(columns=["x", "median", "q25", "q75", "n"])

    ogf = df["OGF"].to_numpy(dtype=float)
    lat = df["latency_sec"].to_numpy(dtype=float)

    # Robust max range (avoid a single extreme bin stretching everything)
    max_ogf = float(np.nanquantile(ogf, 0.99))
    max_ogf = max(max_ogf, float(np.nanmax(ogf)))
    max_ogf = float(np.clip(max_ogf, 0.0, max_ogf))

    if max_ogf <= 0:
        return pd.DataFrame(columns=["x", "median", "q25", "q75", "n"])

    step = ogf_step if max_ogf <= 6 else max(0.5, ogf_step * 2)
    edges = np.arange(0.0, max_ogf + step, step)
    if edges.size < 3:
        edges = np.array([0.0, max_ogf / 2, max_ogf])

    bins = pd.cut(df["OGF"], bins=edges, include_lowest=True, right=False)
    g = df.groupby(bins, observed=True)["latency_sec"]

    rows = []
    for interval, s in g:
        s = s.dropna()
        n = int(s.shape[0])
        if n < min_n:
            continue
        x = float(interval.left + (interval.right - interval.left) / 2.0)
        rows.append({
            "x": x,
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "n": n,
        })
    return pd.DataFrame(rows).sort_values("x")


def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    """
    Mean and ~95% CI half-width using normal approx.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    mu = float(x.mean())
    if x.size == 1:
        return mu, 0.0
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    return mu, 1.96 * se


def rate_ci(p: float, n: int) -> Tuple[float, float]:
    """
    Proportion and ~95% CI half-width (normal approx).
    """
    if n <= 0:
        return 0.0, 0.0
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    return p, 1.96 * se


def compute_overview_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Returns dict(metric -> (value, ci_halfwidth)), with:
      avg_ogf, succ_ge2_pct, stall_pct, mean_latency
    """
    ogf = df["OGF"].to_numpy(dtype=float)
    lat = df["latency_sec"].to_numpy(dtype=float)
    stall = df["stall"].to_numpy(dtype=int) if "stall" in df.columns else np.zeros_like(ogf, dtype=int)

    ogf_mu, ogf_ci = mean_ci(ogf)
    lat_mu, lat_ci = mean_ci(lat)

    n = ogf.size
    succ = float((ogf >= 2.0).mean()) if n else 0.0
    succ_p, succ_ci = rate_ci(succ, int(n))

    st = float((stall == 1).mean()) if n else 0.0
    st_p, st_ci = rate_ci(st, int(n))

    return {
        "avg_ogf": (ogf_mu, ogf_ci),
        "succ_ge2_pct": (succ_p * 100.0, succ_ci * 100.0),
        "stall_pct": (st_p * 100.0, st_ci * 100.0),
        "mean_latency": (lat_mu, lat_ci),
    }




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True,
                    help="Directory containing RL-GOAL evaluation CSV chunks/merged files.")
    ap.add_argument("--out_dir", type=str, default=".",
                    help="Output directory for figures.")
    ap.add_argument("--make_rlgoal", type=str, default="",
                    help="Optional path to make_rlgoal.py to parse its group list.")
    ap.add_argument("--exclude_tags", type=str, default="",
                    help="Comma-separated model tags to exclude (e.g., ds).")
    ap.add_argument("--include_tags", type=str, default="",
                    help="Comma-separated allowlist of model tags to include (if set).")
    ap.add_argument("--max_jitter", type=int, default=1200,
                    help="Max jitter points per condition per model (subsample).")

    args = ap.parse_args()
    in_dir = Path(os.path.expandvars(os.path.expanduser(args.in_dir)))
    out_dir = Path(os.path.expandvars(os.path.expanduser(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"--in_dir not found: {in_dir}")

    exclude = {_clean_tag(t) for t in args.exclude_tags.split(",") if t.strip()}
    include = {_clean_tag(t) for t in args.include_tags.split(",") if t.strip()}

    specs: List[GroupSpec] = []
    if args.make_rlgoal:
        mr = Path(os.path.expandvars(os.path.expanduser(args.make_rlgoal)))
        if mr.exists():
            specs = parse_groups_from_make_rlgoal(mr)

    if not specs:
        specs = scan_groups_from_dir(in_dir)

    # Load trial data for each group and assemble per model/condition
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for gs in specs:
        tag_clean = _clean_tag(gs.model_tag)
        if exclude and tag_clean in exclude:
            continue
        if include and tag_clean not in include:
            continue
        if gs.condition not in {"RL-GOAL", "No-prefix"}:
            continue

        paths = resolve_group_files(in_dir, gs.prefix)
        if not paths:
            continue
        df = load_trials(paths)
        if df.empty:
            continue

        model = gs.model_name
        all_data.setdefault(model, {})
        all_data[model][gs.condition] = df

    if not all_data:
        raise RuntimeError("No usable trial data found. Check --in_dir and file naming.")

    # Matplotlib global style (minimal, paper-friendly)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
    })

    plot_latency_distribution(all_data, out_dir, max_jitter=args.max_jitter)
    plot_tradeoff(all_data, out_dir)
    plot_overview_metrics(all_data, out_dir)

    print("Saved figures to:", out_dir.resolve())
    print(" - rlgoal_latency_dist_B.[pdf|png]")
    print(" - rlgoal_tradeoff_A.[pdf|png]")
    print(" - rlgoal_overview_D.[pdf|png]")


if __name__ == "__main__":
    main()
