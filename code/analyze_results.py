#!/usr/bin/env python3
"""
Phase 3 — Bootstrap CI analysis of multi-seed experiment results
================================================================

Reads results/metrics_multiseed.csv (15 rows: 5 seeds × 3 models) and:

1. Computes per-model mean ± 95% bootstrap CI for all metrics
2. Computes pairwise model deltas (Coord - Original, Coord - SE) with CI
3. Generates a markdown results table ready to paste into this paper §4 Results
4. Generates a Python dict for downstream plot scripts

Output:
  results/analysis_summary.md       (markdown tables ready for the paper)
  results/analysis_summary.json     (machine-readable for plots)

Usage:
    python analyze_results.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_PATH = RESULTS_DIR / "metrics_multiseed.csv"
OUT_MD = RESULTS_DIR / "analysis_summary.md"
OUT_JSON = RESULTS_DIR / "analysis_summary.json"

# Reproducibility for bootstrap sampling
BOOTSTRAP_SEED = 42
N_BOOT = 10000
CONF = 0.95

# Metric display config: (csv_key, display_name, is_percentage)
METRICS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall (Sensitivity)", True),
    ("specificity", "Specificity", True),
    ("f1", "F1-Score", True),
    ("auc_roc", "AUC-ROC", False),
    ("auc_pr", "AUC-PR", False),
]

MODEL_DISPLAY = {
    "original": "Original CNN",
    "se": "SE-Attn CNN",
    "coord": "Coord-Attn CNN",
}


def bootstrap_ci(values, n_iter=N_BOOT, conf=CONF, rng=None):
    """Percentile bootstrap CI for the mean of a small sample."""
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1 - conf
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return values.mean(), lower, upper, values.std(ddof=1)


def bootstrap_diff_ci(values_a, values_b, n_iter=N_BOOT, conf=CONF, rng=None):
    """Bootstrap CI for the mean difference (values_a - values_b).

    Pairs same-seed values, then resamples paired differences.
    """
    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b)
    diffs = a - b
    diff_means = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.choice(len(diffs), size=len(diffs), replace=True)
        diff_means[i] = diffs[idx].mean()
    alpha = 1 - conf
    lower = np.percentile(diff_means, alpha / 2 * 100)
    upper = np.percentile(diff_means, (1 - alpha / 2) * 100)
    return diffs.mean(), lower, upper


def load_results(csv_path=CSV_PATH):
    """Return dict: model_name → list of dicts (one per seed)."""
    grouped = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["model"]].append(row)
    # Sort within each model by seed for stability
    for m in grouped:
        grouped[m].sort(key=lambda r: int(r["seed"]))
    return grouped


def fmt_mean_ci(mean, lo, hi, is_pct=True):
    """Format 'mean (CI_low, CI_high)'."""
    if is_pct:
        return f"{mean*100:.2f} ({lo*100:.2f}, {hi*100:.2f})"
    else:
        return f"{mean:.4f} ({lo:.4f}, {hi:.4f})"


def fmt_diff(mean, lo, hi, is_pct=True):
    """Format signed mean difference + CI."""
    sign = "+" if mean >= 0 else ""
    if is_pct:
        return f"{sign}{mean*100:.2f} ({sign if lo >= 0 else ''}{lo*100:.2f}, {sign if hi >= 0 else ''}{hi*100:.2f}) pp"
    else:
        return f"{sign}{mean:.4f} ({sign if lo >= 0 else ''}{lo:.4f}, {sign if hi >= 0 else ''}{hi:.4f})"


def main():
    grouped = load_results()
    print(f"Loaded {sum(len(v) for v in grouped.values())} rows")
    print(f"Models: {list(grouped.keys())}")
    print(f"Seeds per model: {[len(grouped[m]) for m in grouped]}")

    # Per-model bootstrap CI for each metric
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_model = {}
    for m, rows in grouped.items():
        per_model[m] = {}
        for key, _display, _pct in METRICS:
            values = [float(r[key]) for r in rows]
            mean, lo, hi, std = bootstrap_ci(values, rng=rng)
            per_model[m][key] = {
                "values": values,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "std": std,
            }

    # Pairwise differences (paired by seed)
    pair_diffs = {}
    for a, b in [("coord", "original"), ("coord", "se"), ("se", "original")]:
        pair_diffs[f"{a}_minus_{b}"] = {}
        for key, _display, _pct in METRICS:
            va = per_model[a][key]["values"]
            vb = per_model[b][key]["values"]
            mean, lo, hi = bootstrap_diff_ci(va, vb, rng=rng)
            pair_diffs[f"{a}_minus_{b}"][key] = {
                "mean_diff": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "significant": (lo > 0) or (hi < 0),
            }

    # Build markdown
    lines = []
    lines.append("# This paper — Multi-Seed Bootstrap Analysis\n")
    lines.append(
        f"5 random seeds: 42, 123, 456, 789, 2026  \n"
        f"Bootstrap iterations: {N_BOOT:,}  \n"
        f"Confidence level: {CONF*100:.0f}%  \n"
        f"Test split: stratified 20% per seed (591 samples)\n"
    )
    lines.append(
        "All values reported as **mean (95% CI lower, 95% CI upper)**.\n"
    )

    # Table 1: per-model results
    lines.append("\n## Table 1 — Per-Model Performance (5-seed bootstrap CI)\n")
    headers = ["Metric"] + [MODEL_DISPLAY[m] for m in ["original", "se", "coord"]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for key, display, is_pct in METRICS:
        row = [display]
        for m in ["original", "se", "coord"]:
            r = per_model[m][key]
            row.append(fmt_mean_ci(r["mean"], r["ci_lower"], r["ci_upper"], is_pct))
        lines.append("| " + " | ".join(row) + " |")

    # Parameter count row
    params_per_model = {
        m: int(grouped[m][0]["num_params"]) for m in grouped
    }
    lines.append(
        "| Parameters (fixed) | "
        + " | ".join(
            f"{params_per_model[m]:,}" for m in ["original", "se", "coord"]
        )
        + " |"
    )

    # Table 2: Coord-Attn vs Original (paired difference)
    lines.append(
        "\n## Table 2 — Coord-Attn CNN vs Original CNN (paired bootstrap, same seed)\n"
    )
    lines.append(
        "Values shown as **mean difference (95% CI)**. "
        "Asterisk (\\*) marks differences whose 95% CI excludes zero "
        "(statistically distinguishable from no improvement at α = 0.05).\n"
    )
    lines.append("| Metric | Coord − Original | Significant? |")
    lines.append("|---|---|---|")
    for key, display, is_pct in METRICS:
        d = pair_diffs["coord_minus_original"][key]
        sig = "✓\\*" if d["significant"] else "—"
        lines.append(
            f"| {display} | {fmt_diff(d['mean_diff'], d['ci_lower'], d['ci_upper'], is_pct)} | {sig} |"
        )

    # Table 3: Coord-Attn vs SE-Attn (paired difference)
    lines.append(
        "\n## Table 3 — Coord-Attn CNN vs SE-Attn CNN (paired bootstrap, same seed)\n"
    )
    lines.append("| Metric | Coord − SE | Significant? |")
    lines.append("|---|---|---|")
    for key, display, is_pct in METRICS:
        d = pair_diffs["coord_minus_se"][key]
        sig = "✓\\*" if d["significant"] else "—"
        lines.append(
            f"| {display} | {fmt_diff(d['mean_diff'], d['ci_lower'], d['ci_upper'], is_pct)} | {sig} |"
        )

    # Raw seed-level results
    lines.append("\n## Appendix — Raw Per-Seed Results\n")
    lines.append("| Seed | Model | Accuracy | F1 | AUC-ROC | Sens | Spec |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in ["original", "se", "coord"]:
        for r in grouped[m]:
            lines.append(
                f"| {r['seed']} | {MODEL_DISPLAY[m]} | "
                f"{float(r['accuracy'])*100:.2f}% | "
                f"{float(r['f1'])*100:.2f}% | "
                f"{float(r['auc_roc']):.4f} | "
                f"{float(r['recall'])*100:.2f}% | "
                f"{float(r['specificity'])*100:.2f}% |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ Markdown summary written: {OUT_MD}")

    # JSON for plot scripts
    OUT_JSON.write_text(
        json.dumps(
            {"per_model": per_model, "pair_diffs": pair_diffs,
             "params": params_per_model, "config": {
                 "seeds": [42, 123, 456, 789, 2026],
                 "bootstrap_iters": N_BOOT,
                 "confidence": CONF,
             }},
            indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        ),
        encoding="utf-8",
    )
    print(f"✓ JSON dump written: {OUT_JSON}")

    # Print abbreviated table to console
    print("\n=== Highlights ===\n")
    for key, display, is_pct in METRICS:
        line = f"{display:>22s}: "
        for m in ["original", "se", "coord"]:
            r = per_model[m][key]
            if is_pct:
                line += f"{MODEL_DISPLAY[m][:5]}: {r['mean']*100:.2f}% [{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}]   "
            else:
                line += f"{MODEL_DISPLAY[m][:5]}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]   "
        print(line)


if __name__ == "__main__":
    main()
