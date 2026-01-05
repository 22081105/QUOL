#!/usr/bin/env python3
"""
plot_rlgoal_figures_v4.py

ACL-ready visualization script for RL-GOAL results.
- Loads per-run CSVs (RL-GOAL vs No-prefix) across multiple victim models
- Produces three figures (choose variants A–D per figure):
    1) Latency distribution per model (log y)
    2) OGF–latency tradeoff per model (log y)
    3) Overview metrics (Avg OGF, Success@OGF>=2, Stall%, Mean latency)

Designed to be robust to column-name differences and to work directly from a
`make_rlgoal.py`-style FILES list if available.

Usage (recommended):
  python plot_rlgoal_figures_v4.py \
    --make_rlgoal /path/to/make_rlgoal.py \
    --out_dir /path/to/figs \
    --size double \
    --dist_variant A \
    --tradeoff_variant A \
    --overview_variant A

Alternative (manual specs):
  python plot_rlgoal_figures_v4.py \
    --spec "LLaMA-2-7B,RL-GOAL,/path/to/rlgoal.csv" \
    --spec "LLaMA-2-7B,No-prefix,/path/to/noprefix.csv" \
    --spec "Phi-3-mini,RL-GOAL,/path/to/phi_rl.csv" \
    --spec "Phi-3-mini,No-prefix,/path/to/phi_np.csv" \
    --out_dir figs

Notes:
- This script does NOT generate the methodology schematic.
- Outputs both PDF (vector) + PNG for each figure.
"""

import argparse
import importlib.util
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Palettes (colorblind-friendly)
# -----------------------------
PALETTES: Dict[str, Dict[str, str]] = {
    # Okabe-Ito (commonly used, colorblind-safe)
    "okabe_ito": {
        "No-prefix": "#E69F00",  # orange
        "RL-GOAL":   "#0072B2",  # blue
    },
    # Paul Tol Bright (good contrast)
    "tol_bright": {
        "No-prefix": "#EE6677",  # red-ish
        "RL-GOAL":   "#4477AA",  # blue
    },
    # Minimalist: gray baseline + deep blue RL
    "accent_gray": {
        "No-prefix": "#9AA0A6",
        "RL-GOAL":   "#1f4e79",
    },
    # Paul Tol Muted (paper-friendly)
    "tol_muted": {
        "No-prefix": "#CCBB44",
        "RL-GOAL":   "#332288",
    },
}

# Map variant -> palette (mirrors the sample sheet)
DEFAULT_PALETTE_FOR_VARIANT = {
    "A": "okabe_ito",
    "B": "tol_bright",
    "C": "accent_gray",
    "D": "tol_muted",
}


# -----------------------------
# Styling
# -----------------------------
def apply_rc(font: str = "serif") -> None:
    """
    Paper-friendly matplotlib defaults.
    Use 'serif' to better match ACL templates; fallback is safe.
    """
    base = {
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "lines.linewidth": 2.2,
        "lines.markersize": 4.0,
    }
    plt.rcParams.update(base)
    if font == "serif":
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "stix",
        })


# -----------------------------
# Robust CSV ingestion
# -----------------------------
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm:
            return norm[key]
    # try fuzzy contains
    for cand in candidates:
        key = _norm_col(cand)
        for k, orig in norm.items():
            if key and key in k:
                return orig
    return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize common columns into canonical names (if present)
    ogf_col = pick_col(df, ["OGF", "ogf", "max_ogf", "maxOGF"])
    lat_col = pick_col(df, ["latency_sec", "latency (s)", "latency_s", "latency", "Latency", "Latency (s)"])
    stall_col = pick_col(df, ["stall", "stall_flag", "Stall", "stall%"])
    # EOS presence sometimes recorded; allow later derivation if needed
    eos_col = pick_col(df, ["has_eos", "eos", "eos_seen", "EOS"])
    cap_col = pick_col(df, ["hit_cap", "cap_hit", "hit_generation_cap", "gen_cap_hit"])

    out = pd.DataFrame()
    if ogf_col is not None:
        out["OGF"] = coerce_numeric(df[ogf_col])
    if lat_col is not None:
        out["latency_sec"] = coerce_numeric(df[lat_col])
    if stall_col is not None:
        out["stall"] = coerce_numeric(df[stall_col])
    if eos_col is not None:
        out["has_eos"] = coerce_numeric(df[eos_col])
    if cap_col is not None:
        out["hit_cap"] = coerce_numeric(df[cap_col])

    # Keep any helpful context columns if present (for debugging)
    for extra in ["model", "setting", "seed", "trial", "prompt_id", "prompt_idx"]:
        c = pick_col(df, [extra])
        if c is not None and c not in out.columns:
            out[extra] = df[c]

    # Derive stall if not present but eos/cap present:
    if "stall" not in out.columns:
        if ("has_eos" in out.columns) and ("hit_cap" in out.columns):
            out["stall"] = ((out["has_eos"] <= 0.0) & (out["hit_cap"] > 0.0)).astype(float)

    return out


def load_from_make_rlgoal(make_rlgoal_py: Path) -> List[Tuple[str, str, Path]]:
    """
    Expects a variable named FILES with entries like:
      (model_name, setting_name, path_to_csv)
    setting_name may be 'rlgoal'/'noprefix'/etc.
    """
    spec = importlib.util.spec_from_file_location("make_rlgoal_module", str(make_rlgoal_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {make_rlgoal_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    files = getattr(mod, "FILES", None)
    if not files:
        raise RuntimeError("make_rlgoal.py does not define FILES (or it's empty).")

    triples: List[Tuple[str, str, Path]] = []
    for item in files:
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        model, setting, path = item[0], item[1], item[2]
        model = str(model).strip()
        setting_raw = str(setting).strip().lower()
        if "rl" in setting_raw:
            setting_nice = "RL-GOAL"
        elif "no" in setting_raw or "base" in setting_raw or "noprefix" in setting_raw:
            setting_nice = "No-prefix"
        else:
            # fallback: keep but label
            setting_nice = str(setting)
        triples.append((model, setting_nice, Path(path)))
    return triples


def load_specs(spec_args: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Each spec: "MODEL,SETTING,PATH"
    """
    out: List[Tuple[str, str, Path]] = []
    for s in spec_args:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad --spec '{s}'. Expected: MODEL,SETTING,PATH")
        model, setting, path = parts
        if setting.lower() in ["rlgoal", "rl-goal", "rl_goal"]:
            setting = "RL-GOAL"
        if setting.lower() in ["noprefix", "no-prefix", "baseline", "no_prefix"]:
            setting = "No-prefix"
        out.append((model, setting, Path(path)))
    return out


def load_all(triples: List[Tuple[str, str, Path]]) -> pd.DataFrame:
    frames = []
    for model, setting, path in triples:
        if not path.exists():
            print(f"[WARN] Missing file, skipping: {path}")
            continue
        df = load_one_csv(path)
        if "OGF" not in df.columns or "latency_sec" not in df.columns:
            print(f"[WARN] {path} missing OGF/latency columns after parsing; skipping.")
            continue
        df["model"] = model
        df["setting"] = setting
        frames.append(df)
    if not frames:
        raise RuntimeError("No usable CSVs were loaded.")
    out = pd.concat(frames, ignore_index=True)

    # Drop unusable rows
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["OGF", "latency_sec"])
    out = out[(out["OGF"] >= 0) & (out["latency_sec"] > 0)]
    return out


# -----------------------------
# Stats helpers
# -----------------------------
def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / max(len(x), 1)
    return x, y


def robust_bin_summary(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.digitize(x, bins) - 1
    med = np.full(len(bins) - 1, np.nan)
    lo = np.full(len(bins) - 1, np.nan)
    hi = np.full(len(bins) - 1, np.nan)
    xc = 0.5 * (bins[:-1] + bins[1:])
    for b in range(len(bins) - 1):
        m = (idx == b)
        if m.sum() < 30:
            continue
        ys = y[m]
        med[b] = np.median(ys)
        lo[b] = np.quantile(ys, 0.25)
        hi[b] = np.quantile(ys, 0.75)
    return xc, med, lo, hi


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, setting), s in df.groupby(["model", "setting"]):
        ogf = s["OGF"].to_numpy()
        lat = s["latency_sec"].to_numpy()
        stall = None
        if "stall" in s.columns:
            stall = s["stall"].to_numpy()
        rows.append({
            "model": model,
            "setting": setting,
            "avg_ogf": float(np.mean(ogf)),
            "succ_ge2": float(np.mean(ogf >= 2.0) * 100.0),
            "stall_pct": float(np.mean(stall > 0.0) * 100.0) if stall is not None else np.nan,
            "mean_lat": float(np.mean(lat)),
        })
    out = pd.DataFrame(rows)

    # Ensure stable order for plotting
    # (If you want custom ordering, just pass --model_order.)
    return out


# -----------------------------
# Figure sizes (ACL-friendly)
# -----------------------------
def figsize_for(size: str) -> Tuple[float, float]:
    # ACL single column ~ 3.35in, double column ~ 7in
    if size == "single":
        return (3.35, 2.4)
    return (7.0, 4.9)


# -----------------------------
# Plot 1: latency distribution
# -----------------------------
def plot_latency_distribution(df: pd.DataFrame, variant: str, palette: Dict[str, str], size: str) -> plt.Figure:
    models = list(df["model"].drop_duplicates())
    n = len(models)
    ncols = 2 if size == "double" else 1
    nrows = int(math.ceil(n / ncols))
    fig_w, fig_h = figsize_for(size)
    fig_h = fig_h * (nrows / 2.0) if size == "double" else fig_h * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)

    for i, model in enumerate(models):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        sub = df[df["model"] == model]

        if variant == "A":  # Violin + median/IQR marker
            data = [
                sub[sub["setting"] == "No-prefix"]["latency_sec"].to_numpy(),
                sub[sub["setting"] == "RL-GOAL"]["latency_sec"].to_numpy(),
            ]
            parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.86)
            for j, body in enumerate(parts["bodies"]):
                setting = "No-prefix" if j == 0 else "RL-GOAL"
                body.set_facecolor(palette.get(setting, "#777777"))
                body.set_edgecolor("none")
                body.set_alpha(0.28)
            for j, setting in enumerate(["No-prefix", "RL-GOAL"], start=1):
                vals = sub[sub["setting"] == setting]["latency_sec"].to_numpy()
                q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
                ax.plot([j, j], [q1, q3], color="black", lw=1.2, solid_capstyle="round")
                ax.scatter([j], [med], color="black", s=24, zorder=3)
            ax.set_xticks([1, 2], ["No-prefix", "RL-GOAL"])

        elif variant == "B":  # box + light jitter
            # jitter sample for readability
            for j, setting in enumerate(["No-prefix", "RL-GOAL"], start=1):
                vals = sub[sub["setting"] == setting]["latency_sec"].to_numpy()
                if len(vals) == 0:
                    continue
                samp = vals[np.random.default_rng(0).choice(len(vals), size=min(700, len(vals)), replace=False)]
                xj = j + np.random.default_rng(0).normal(0, 0.055, size=len(samp))
                ax.scatter(xj, samp, s=6, alpha=0.12, color=palette.get(setting, "#777777"), edgecolors="none")
            bp = ax.boxplot(
                [
                    sub[sub["setting"] == "No-prefix"]["latency_sec"].to_numpy(),
                    sub[sub["setting"] == "RL-GOAL"]["latency_sec"].to_numpy(),
                ],
                positions=[1, 2],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", lw=1.15),
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )
            for patch, setting in zip(bp["boxes"], ["No-prefix", "RL-GOAL"]):
                patch.set_facecolor(palette.get(setting, "#777777"))
                patch.set_alpha(0.22)
            ax.set_xticks([1, 2], ["No-prefix", "RL-GOAL"])

        elif variant == "C":  # ECDF
            for setting in ["No-prefix", "RL-GOAL"]:
                vals = sub[sub["setting"] == setting]["latency_sec"].to_numpy()
                x, y = ecdf(vals)
                ax.plot(x, y, lw=2.2, color=palette.get(setting, "#777777"), label=setting)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("Latency (s)")
            ax.set_ylabel("ECDF")
            ax.legend(frameon=False, loc="lower right")
            ax.grid(True, which="both")

        elif variant == "D":  # log-hist overlay
            bins = np.logspace(-2, 4, 42)
            for setting in ["No-prefix", "RL-GOAL"]:
                vals = sub[sub["setting"] == setting]["latency_sec"].to_numpy()
                ax.hist(vals, bins=bins, density=True, alpha=0.28, color=palette.get(setting, "#777777"), label=setting)
            ax.set_xscale("log")
            ax.set_ylabel("Density")
            ax.legend(frameon=False, loc="upper right")
            ax.grid(True, which="both")

        ax.set_title(model)
        ax.set_yscale("log")
        ax.set_ylabel("Latency (s)")
    return fig


# -----------------------------
# Plot 2: OGF–latency tradeoff
# -----------------------------
def plot_tradeoff(df: pd.DataFrame, variant: str, palette: Dict[str, str], size: str) -> plt.Figure:
    models = list(df["model"].drop_duplicates())
    n = len(models)
    ncols = 2 if size == "double" else 1
    nrows = int(math.ceil(n / ncols))
    fig_w, fig_h = figsize_for(size)
    fig_h = fig_h * (nrows / 2.0) if size == "double" else fig_h * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)

    for i, model in enumerate(models):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        sub = df[df["model"] == model]
        xmax = float(np.quantile(sub["OGF"].to_numpy(), 0.98))
        xmax = max(1.5, xmax)
        bins = np.linspace(0, xmax, 14)

        if variant == "A":  # binned median + IQR band
            for setting in ["No-prefix", "RL-GOAL"]:
                s = sub[sub["setting"] == setting]
                xc, med, lo, hi = robust_bin_summary(s["OGF"].to_numpy(), s["latency_sec"].to_numpy(), bins=bins)
                ax.fill_between(xc, lo, hi, color=palette.get(setting, "#777777"), alpha=0.18, linewidth=0)
                ax.plot(xc, med, color=palette.get(setting, "#777777"), lw=2.4, marker="o", ms=3.6, label=setting)

        elif variant == "B":  # scatter subsample + median curve
            rng = np.random.default_rng(0)
            for setting in ["No-prefix", "RL-GOAL"]:
                s = sub[sub["setting"] == setting]
                n_s = min(2500, len(s))
                ss = s.sample(n=n_s, random_state=0) if len(s) > n_s else s
                ax.scatter(ss["OGF"], ss["latency_sec"], s=10, alpha=0.10, color=palette.get(setting, "#777777"), edgecolors="none")
                xc, med, lo, hi = robust_bin_summary(s["OGF"].to_numpy(), s["latency_sec"].to_numpy(), bins=bins)
                ax.plot(xc, med, color=palette.get(setting, "#777777"), lw=2.6, label=setting)

        elif variant == "C":  # quantile fan (10/50/90)
            for setting in ["No-prefix", "RL-GOAL"]:
                s = sub[sub["setting"] == setting]
                idx = np.digitize(s["OGF"].to_numpy(), bins) - 1
                xc = 0.5 * (bins[:-1] + bins[1:])
                q10 = np.full(len(bins) - 1, np.nan)
                q50 = np.full(len(bins) - 1, np.nan)
                q90 = np.full(len(bins) - 1, np.nan)
                for b in range(len(bins) - 1):
                    m = (idx == b)
                    if m.sum() < 30:
                        continue
                    ys = s.loc[m, "latency_sec"].to_numpy()
                    q10[b], q50[b], q90[b] = np.quantile(ys, [0.10, 0.50, 0.90])
                ax.fill_between(xc, q10, q90, color=palette.get(setting, "#777777"), alpha=0.12, linewidth=0)
                ax.plot(xc, q50, color=palette.get(setting, "#777777"), lw=2.6, marker="o", ms=3.6, label=setting)

        elif variant == "D":  # ECDF over OGF buckets (compact)
            # Show how mass shifts to higher OGF by plotting ECDF of OGF, with latency-coded markers.
            # (Useful when you want to emphasize distributional shift rather than correlation.)
            for setting in ["No-prefix", "RL-GOAL"]:
                ogf = sub[sub["setting"] == setting]["OGF"].to_numpy()
                x, y = ecdf(ogf)
                ax.plot(x, y, lw=2.4, color=palette.get(setting, "#777777"), label=setting)
            ax.set_ylabel("ECDF(OGF)")
            ax.set_ylim(0, 1.0)

        ax.set_title(model)
        ax.set_xlabel("OGF")
        ax.set_ylabel("Latency (s)")
        if variant != "D":
            ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.legend(frameon=False, loc="lower right")
    return fig


# -----------------------------
# Plot 3: overview metrics
# -----------------------------
def plot_overview(sumdf: pd.DataFrame, variant: str, palette: Dict[str, str], size: str, model_order: Optional[List[str]] = None) -> plt.Figure:
    if model_order is None:
        order = list(sumdf["model"].drop_duplicates())
    else:
        order = model_order

    def get(model: str, setting: str, col: str) -> float:
        v = sumdf[(sumdf["model"] == model) & (sumdf["setting"] == setting)][col]
        return float(v.values[0]) if len(v) else float("nan")

    if variant in ["A", "B"]:
        fig_w, fig_h = figsize_for(size)
        fig_h = 4.4 if size == "double" else 4.8
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
        axes = axes.ravel()

        metrics = [
            ("avg_ogf", "Avg. OGF", False),
            ("succ_ge2", "Success @ OGF≥2 (%)", False),
            ("stall_pct", "Stall rate (%)", False),
            ("mean_lat", "Mean latency (s)", True),
        ]
        for ax, (col, title, logy) in zip(axes, metrics):
            for i, m in enumerate(order):
                a = get(m, "No-prefix", col)
                b = get(m, "RL-GOAL", col)
                ax.plot([i, i], [a, b], color="#9A9A9A", lw=1.8, zorder=1)
                ax.scatter([i], [a], s=55, color=palette.get("No-prefix", "#777777"), edgecolors="white", linewidths=0.8, zorder=3)
                ax.scatter([i], [b], s=55, color=palette.get("RL-GOAL", "#777777"), edgecolors="white", linewidths=0.8, zorder=3)
            ax.set_title(title)
            ax.set_xticks(range(len(order)), order, rotation=18, ha="right")
            if logy:
                ax.set_yscale("log")
            ax.grid(True, axis="y")

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get("No-prefix", "#777777"),
                       markeredgecolor="white", markeredgewidth=0.8, markersize=8, label="No-prefix"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get("RL-GOAL", "#777777"),
                       markeredgecolor="white", markeredgewidth=0.8, markersize=8, label="RL-GOAL"),
        ]
        fig.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04))
        return fig

    if variant == "C":
        # Heatmap of deltas (RL - No-prefix); latency uses × ratio
        metrics = [("avg_ogf", "Avg.OGF"), ("succ_ge2", "S@≥2"), ("stall_pct", "Stall"), ("mean_lat", "MeanLat×")]
        delta = np.zeros((len(order), len(metrics)))
        for i, m in enumerate(order):
            for j, (col, _) in enumerate(metrics):
                a = get(m, "No-prefix", col)
                b = get(m, "RL-GOAL", col)
                if col == "mean_lat":
                    delta[i, j] = (b / max(a, 1e-9))
                else:
                    delta[i, j] = (b - a)

        fig_w, _ = figsize_for(size)
        fig, ax = plt.subplots(figsize=(fig_w, 2.2), constrained_layout=True)
        im = ax.imshow(delta, aspect="auto")
        ax.set_yticks(range(len(order)), order)
        ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
        ax.set_title("Δ (RL-GOAL − No-prefix); latency is × ratio")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f"{delta[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        return fig

    if variant == "D":
        # 1×4 small-multiples paired bars (compact; less preferred than A/B)
        fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.8), constrained_layout=True)
        metrics = [
            ("avg_ogf", "Avg. OGF", None),
            ("succ_ge2", "S@OGF≥2 (%)", None),
            ("stall_pct", "Stall (%)", None),
            ("mean_lat", "Mean latency (s)", "log"),
        ]
        x = np.arange(len(order))
        w = 0.34
        for ax, (col, title, scale) in zip(axes, metrics):
            a = [get(m, "No-prefix", col) for m in order]
            b = [get(m, "RL-GOAL", col) for m in order]
            ax.bar(x - w / 2, a, width=w, color=palette.get("No-prefix", "#777777"), alpha=0.78, label="No-prefix")
            ax.bar(x + w / 2, b, width=w, color=palette.get("RL-GOAL", "#777777"), alpha=0.78, label="RL-GOAL")
            ax.set_title(title)
            ax.set_xticks(x, order, rotation=18, ha="right")
            if scale == "log":
                ax.set_yscale("log")
            ax.grid(True, axis="y")
        fig.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        return fig

    raise ValueError(f"Unknown overview variant: {variant}")


# -----------------------------
# Output
# -----------------------------
def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--make_rlgoal", type=str, default=None, help="Path to make_rlgoal.py that defines FILES.")
    p.add_argument("--spec", action="append", default=[], help="Manual spec: MODEL,SETTING,PATH (repeatable).")
    p.add_argument("--out_dir", type=str, default="rlgoal_figs", help="Output directory.")
    p.add_argument("--size", choices=["single", "double"], default="double", help="Figure width target (ACL).")
    p.add_argument("--font", choices=["serif", "sans"], default="serif")

    p.add_argument("--dist_variant", choices=["A", "B", "C", "D"], default="A")
    p.add_argument("--tradeoff_variant", choices=["A", "B", "C", "D"], default="A")
    p.add_argument("--overview_variant", choices=["A", "B", "C", "D"], default="A")

    p.add_argument("--palette", choices=list(PALETTES.keys()), default=None,
                   help="Override palette. If omitted, uses a palette tied to the chosen variants.")
    p.add_argument("--model_order", type=str, default=None,
                   help="Comma-separated model order (optional).")

    args = p.parse_args()

    apply_rc(font=args.font)

    triples: List[Tuple[str, str, Path]] = []
    if args.make_rlgoal:
        triples.extend(load_from_make_rlgoal(Path(args.make_rlgoal)))
    triples.extend(load_specs(args.spec))

    if not triples:
        raise SystemExit("No inputs. Provide --make_rlgoal or at least one --spec.")

    df = load_all(triples)

    # Stable ordering (optional)
    model_order = None
    if args.model_order:
        model_order = [m.strip() for m in args.model_order.split(",") if m.strip()]

    # Choose palette
    if args.palette:
        pal = PALETTES[args.palette]
    else:
        # If no override, pick palette matching dist variant (consistent across figs)
        pal = PALETTES[DEFAULT_PALETTE_FOR_VARIANT.get(args.dist_variant, "okabe_ito")]

    out_dir = Path(args.out_dir)

    # 1) Latency distribution
    fig1 = plot_latency_distribution(df, args.dist_variant, pal, args.size)
    save_both(fig1, out_dir, f"rlgoal_latency_dist_{args.dist_variant}")

    # 2) Tradeoff
    fig2 = plot_tradeoff(df, args.tradeoff_variant, pal, args.size)
    save_both(fig2, out_dir, f"rlgoal_tradeoff_{args.tradeoff_variant}")

    # 3) Overview
    sumdf = summarise(df)
    fig3 = plot_overview(sumdf, args.overview_variant, pal, args.size, model_order=model_order)
    save_both(fig3, out_dir, f"rlgoal_overview_{args.overview_variant}")

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
