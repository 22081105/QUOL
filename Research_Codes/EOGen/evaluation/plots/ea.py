#!/usr/bin/env python3
# ea.py
#
# Make 3 plots (each overlays 3 models) from EOGen chunk CSVs:
#   1) Histogram of per-prompt Success@OGF>=1 (mean line per model)
#   2) Histogram of per-prompt max OGF (max over trials) with OGF threshold markers
#   3) CDF of per-prompt stall rates with pooled threshold annotations
#
# Example:
# cd Research_Codes/EOGen/evaluation/plots
# python ea.py \
#   --phi_dir   /Research_Codes/EOGen/evaluation/Phi-3-mini-4k-instruct/test_eogen \
#   --llama_dir /Research_Codes/EOGen/evaluation/Llama-2-7b-hf/test_eogen \
#   --ds_dir    /Research_Codes/EOGen/evaluation/deepseek-coder-7b-base-v1.5/test_eogen \
#   --pattern   "eogen_results_*chunk*_of_15.csv" \
#   --out_dir   /comb

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Headless-friendly backend for HPC
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

# Paper-friendly defaults + modern, colorblind-safe palette (Paul Tol "bright"-style)
PAPER_COLORS = ["#4477AA", "#EE6677", "#228833"]  # blue, pink/red, green

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,  # editable text in PDF
    "ps.fonttype": 42,
    "axes.prop_cycle": matplotlib.cycler(color=PAPER_COLORS),
})


@dataclass
class ModelAgg:
    label: str
    prompt_success: list[float]   # per-prompt success rate
    prompt_maxogf: list[float]    # per-prompt max OGF
    prompt_stall: list[float]     # per-prompt stall rate


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def list_csvs(d: Path, pattern: str) -> list[Path]:
    if not d.exists():
        return []
    return sorted(d.glob(pattern))


def aggregate_model(model_dir: Path, pattern: str, label: str | None = None) -> ModelAgg:
    """
    Aggregates across ALL chunk CSVs in model_dir matching pattern.

    We aggregate from TRIAL rows only:
      per-prompt success rate  = mean(TRIAL.success)
      per-prompt max OGF       = max(TRIAL.OGF)
      per-prompt stall rate    = mean(TRIAL.stall)

    If a per-trial identifier column is present, we dedupe (prompt_key, trial_id).
    """
    csvs = list_csvs(model_dir, pattern)
    out_label = label if label is not None else model_dir.name
    if not csvs:
        return ModelAgg(label=out_label, prompt_success=[], prompt_maxogf=[], prompt_stall=[])

    # prompt_key -> accumulators over TRIAL rows
    n_trials = defaultdict(int)
    sum_success = defaultdict(float)
    sum_stall = defaultdict(float)
    max_ogf = defaultdict(lambda: -1e9)

    # Optional dedupe
    trial_id_fields = ["trial_id", "trial_idx", "trial", "t", "run", "run_id", "sample_id", "seed"]
    seen = set()

    for p in csvs:
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("row_type") != "TRIAL":
                    continue

                pkey = row.get("prompt_key")
                if not pkey:
                    continue

                # Dedupe if we can identify a trial key
                trial_marker = None
                for k in trial_id_fields:
                    v = row.get(k)
                    if v is not None and str(v).strip() != "":
                        trial_marker = str(v).strip()
                        break
                if trial_marker is not None:
                    sig = (pkey, trial_marker)
                    if sig in seen:
                        continue
                    seen.add(sig)

                succ = _safe_float(row.get("success"), default=0.0)
                stl = _safe_float(row.get("stall"), default=0.0)
                ogf = _safe_float(row.get("OGF"), default=None)

                n_trials[pkey] += 1
                sum_success[pkey] += float(succ) if succ is not None else 0.0
                sum_stall[pkey] += float(stl) if stl is not None else 0.0

                if ogf is not None and ogf > max_ogf[pkey]:
                    max_ogf[pkey] = ogf

    prompt_success, prompt_maxogf, prompt_stall = [], [], []
    for pkey, n in n_trials.items():
        if n <= 0:
            continue
        prompt_success.append(sum_success[pkey] / n)
        prompt_stall.append(sum_stall[pkey] / n)
        m = max_ogf[pkey]
        prompt_maxogf.append(m if m > -1e8 else 0.0)

    return ModelAgg(label=out_label, prompt_success=prompt_success, prompt_maxogf=prompt_maxogf, prompt_stall=prompt_stall)


def save_both(fig_basename: str, out_dir: Path):
    out_png = out_dir / f"{fig_basename}.png"
    out_pdf = out_dir / f"{fig_basename}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_png, out_pdf



def plot_success_hist(models: dict[str, ModelAgg], out_dir: Path, normalize: bool):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 11)

    names = list(models.keys())
    m = len(names)
    bin_left = bins[:-1]
    bin_w = np.diff(bins)
    bar_w = 0.86 * bin_w / max(m, 1)  # slightly wider for a clean look

    any_data = False
    for i, (name, agg) in enumerate(models.items()):
        if not agg.prompt_success:
            continue
        any_data = True

        x = np.asarray(agg.prompt_success, dtype=float)
        if normalize:
            weights = np.ones_like(x, dtype=float) / x.size
            hist, _ = np.histogram(x, bins=bins, weights=weights)
        else:
            hist, _ = np.histogram(x, bins=bins)

        offset = (i - (m - 1) / 2.0) * bar_w
        bars = ax.bar(
            bin_left + offset,
            hist,
            width=bar_w,
            align="edge",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.95,
            label=f"{name} (n={x.size})",
        )

        # mean line in matching color
        c = bars.patches[0].get_facecolor() if bars.patches else None
        ax.axvline(float(np.mean(x)), linestyle="--", linewidth=2.2, color=c, alpha=0.9)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Per-prompt Success@OGF $\geq 1$")
    ax.set_ylabel("Fraction of prompts" if normalize else "Number of prompts")
    ax.set_title("Distribution of Per-prompt Success Rates")

    # modern “despine”
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, fontsize=8, loc="upper right")

    return save_both("eogen_success_hist_allmodels", out_dir)


def plot_maxogf_hist(models: dict[str, ModelAgg], out_dir: Path, normalize: bool, ogf_min: float):
    pooled = [x for agg in models.values() for x in agg.prompt_maxogf]
    xmax = max(4.0, float(np.max(pooled))) if pooled else 4.0
    bins_ogf = np.linspace(ogf_min, xmax, 15)

    fig, ax = plt.subplots(figsize=(6, 4))

    names = list(models.keys())
    m = len(names)
    bin_left = bins_ogf[:-1]
    bin_w = np.diff(bins_ogf)
    bar_w = 0.86 * bin_w / max(m, 1)

    any_data = False
    for i, (name, agg) in enumerate(models.items()):
        if not agg.prompt_maxogf:
            continue
        any_data = True

        x = np.asarray(agg.prompt_maxogf, dtype=float)
        if normalize:
            weights = np.ones_like(x, dtype=float) / x.size
            hist, _ = np.histogram(x, bins=bins_ogf, weights=weights)
        else:
            hist, _ = np.histogram(x, bins=bins_ogf)

        offset = (i - (m - 1) / 2.0) * bar_w
        ax.bar(
            bin_left + offset,
            hist,
            width=bar_w,
            align="edge",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.95,
            label=f"{name} (n={x.size})",
        )

    if not any_data:
        plt.close(fig)
        return None



    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax.set_xlabel("Per-prompt max OGF (max length / context window)")
    ax.set_ylabel("Fraction of prompts" if normalize else "Number of prompts")
    ax.set_title("Distribution of Per-prompt Max OGF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, fontsize=8, loc="upper left")
    return save_both("eogen_maxogf_hist_allmodels", out_dir)



def plot_stall_cdf(models: dict[str, ModelAgg], out_dir: Path, thresholds=(0.1, 0.5)):
    plt.figure(figsize=(6, 4))

    any_data = False
    pooled_vals = []
    for name, agg in models.items():
        if not agg.prompt_stall:
            continue
        any_data = True
        x = np.sort(np.asarray(agg.prompt_stall, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=name)
        pooled_vals.append(x)

    if not any_data:
        return None

    # pooled annotations (like your single-model plot)
    if pooled_vals:
        pooled = np.sort(np.concatenate(pooled_vals))
        used_y = []
        for thr in thresholds:
            frac = float(np.mean(pooled <= thr))
            plt.axvline(thr, linestyle=":", linewidth=2)
            plt.axhline(frac, linestyle=":", linewidth=1)

            # label placement: keep away from top edge + avoid overlapping other labels
            if frac > 0.95:
                y = 0.92
                va = "top"
            else:
                y = min(0.98, frac + 0.03)
                va = "bottom"

            for py in used_y:
                if abs(y - py) < 0.06:
                    y = max(0.02, y - 0.08)
                    va = "top"

            used_y.append(y)
            plt.text(thr + 0.01, y, f"{frac*100:.0f}% \u2264 {thr}", fontsize=10, va=va)


    plt.xlabel("Per-prompt stall rate (fraction of runs that hit B without EOS)")
    plt.ylabel("Fraction of prompts")
    plt.title("CDF of Per-prompt Stall Rates")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False, fontsize=8, loc="lower right")

    plt.tight_layout()
    return save_both("eogen_stall_cdf_allmodels", out_dir)


def fmt_stats(label: str, agg: ModelAgg) -> str:
    n = len(agg.prompt_success)
    if n == 0:
        return f"{label}: prompts=0, mean_success=nan, mean_maxOGF=nan, mean_stall=nan"
    mean_success = float(np.mean(agg.prompt_success))
    mean_maxogf = float(np.mean(agg.prompt_maxogf))
    mean_stall = float(np.mean(agg.prompt_stall))
    return (f"{label}: prompts={n}, mean_success={mean_success:.3f}, "
            f"mean_maxOGF={mean_maxogf:.3f}, mean_stall={mean_stall:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", required=True, help="Directory containing Phi-3 chunk CSVs")
    ap.add_argument("--llama_dir", required=True, help="Directory containing LLaMA chunk CSVs")
    ap.add_argument("--ds_dir", required=True, help="Directory containing DeepSeek chunk CSVs")
    ap.add_argument("--pattern", default="eogen_results_*chunk*_of_15.csv", help="Glob pattern for chunk CSVs")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots")

    ap.add_argument("--phi_label", default="Phi-3-mini-4k", help="Legend label for Phi-3")
    ap.add_argument("--llama_label", default="LLaMA-2-7b-hf", help="Legend label for LLaMA")
    ap.add_argument("--ds_label", default="DeepSeek-Coder", help="Legend label for DeepSeek")

    ap.add_argument("--normalize", action="store_true",
                    help="Normalize histograms so each model sums to 1 (y-axis = fraction of prompts).")
    ap.add_argument("--ogf_min", type=float, default=0.5,
                    help="Minimum x for max-OGF histogram bins (default 0.5 to match your style).")
    args = ap.parse_args()

    phi_dir = Path(os.path.expandvars(args.phi_dir)).expanduser().resolve()
    llama_dir = Path(os.path.expandvars(args.llama_dir)).expanduser().resolve()
    ds_dir = Path(os.path.expandvars(args.ds_dir)).expanduser().resolve()
    out_dir = Path(os.path.expandvars(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    phi_agg = aggregate_model(phi_dir, args.pattern, label=args.phi_label)
    llama_agg = aggregate_model(llama_dir, args.pattern, label=args.llama_label)
    ds_agg = aggregate_model(ds_dir, args.pattern, label=args.ds_label)

    models = {
        args.phi_label: phi_agg,
        args.llama_label: llama_agg,
        args.ds_label: ds_agg,
    }

    # Print summary lines
    print(fmt_stats(args.phi_label, phi_agg))
    print(fmt_stats(args.llama_label, llama_agg))
    print(fmt_stats(args.ds_label, ds_agg))

    # Also print matched files (helps debugging)
    print(f"[files] {args.phi_label} matched: {len(list_csvs(phi_dir, args.pattern))} in {phi_dir}")
    print(f"[files] {args.llama_label} matched: {len(list_csvs(llama_dir, args.pattern))} in {llama_dir}")
    print(f"[files] {args.ds_label} matched: {len(list_csvs(ds_dir, args.pattern))} in {ds_dir}")

    r1 = plot_success_hist(models, out_dir, normalize=args.normalize)
    r2 = plot_maxogf_hist(models, out_dir, normalize=args.normalize, ogf_min=args.ogf_min)
    r3 = plot_stall_cdf(models, out_dir)

    if not (r1 or r2 or r3):
        raise SystemExit("No plots produced (no data found). Check --pattern and directories.")

    print(f"Saved 3 plots (pdf+png) to: {out_dir}")


if __name__ == "__main__":
    main()
