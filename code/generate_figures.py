#!/usr/bin/env python3
"""
Phase 4 — Generate This paper figures
==================================

Produces all required figures from multi-seed results:

  fig1_metric_bars.png    — 5-seed mean ± 95% CI bar chart for key metrics
  fig2_roc_curves.png     — ROC curves (one per model, averaged across 5 seeds)
  fig3_pr_curves.png      — Precision-Recall curves (averaged across 5 seeds)
  fig4_f1_vs_threshold.png — F1 vs decision threshold (averaged across 5 seeds)
  fig5_confusion_matrices.png — 3 confusion matrices side by side (median seed)
  fig6_param_efficiency.png  — Accuracy vs Parameter count
  fig7_attention_heatmap.png — CA-1D attention heatmap on a sample apnea window

Output: figures/*.png at 300 DPI
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROBAS_DIR = RESULTS_DIR / "probas"
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
MODELS = ["original", "se", "coord"]
MODEL_DISPLAY = {
    "original": "Original CNN",
    "se": "SE-Attn CNN",
    "coord": "Coord-Attn CNN",
}
COLORS = {
    "original": "#7f7f7f",  # gray
    "se": "#1f77b4",         # blue
    "coord": "#d62728",      # red
}

# Common plot styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_summary():
    return json.loads((RESULTS_DIR / "analysis_summary.json").read_text())


def load_probas(model, seed):
    """Load saved (y_test, y_proba) for one model/seed run."""
    npz = np.load(PROBAS_DIR / f"{model}_seed{seed}.npz")
    return npz["y_test"], npz["y_proba"]


# ============================================================
# FIG 1 — Metric bar chart with 95% CI error bars
# ============================================================
def fig_metric_bars(summary):
    metrics_to_plot = [
        ("accuracy", "Accuracy", True),
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("specificity", "Specificity", True),
        ("f1", "F1-Score", True),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_metrics = len(metrics_to_plot)
    n_models = len(MODELS)
    width = 0.27
    x = np.arange(n_metrics)
    for i, m in enumerate(MODELS):
        means, lows, highs = [], [], []
        for key, _disp, is_pct in metrics_to_plot:
            r = summary["per_model"][m][key]
            mult = 100 if is_pct else 1
            means.append(r["mean"] * mult)
            lows.append(r["mean"] * mult - r["ci_lower"] * mult)
            highs.append(r["ci_upper"] * mult - r["mean"] * mult)
        bars = ax.bar(
            x + (i - 1) * width, means, width,
            yerr=[lows, highs], capsize=4,
            label=MODEL_DISPLAY[m], color=COLORS[m],
            edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        for b, v in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d, _ in metrics_to_plot])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70, 100)
    ax.set_title(
        "Multi-Seed Performance Comparison (5 seeds, 95% bootstrap CI)",
        pad=10,
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_metric_bars.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig1_metric_bars.png")


# ============================================================
# FIG 2 — ROC curves (one curve per model, averaged across seeds)
# ============================================================
def fig_roc_curves(summary):
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for m in MODELS:
        tpr_per_seed = []
        auc_per_seed = []
        for s in SEEDS:
            y_test, y_proba = load_probas(m, s)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tpr_per_seed.append(tpr_interp)
            auc_per_seed.append(auc(fpr, tpr))
        tpr_per_seed = np.array(tpr_per_seed)
        mean_tpr = tpr_per_seed.mean(axis=0)
        std_tpr = tpr_per_seed.std(axis=0)
        mean_auc = np.mean(auc_per_seed)
        ax.plot(
            mean_fpr, mean_tpr, color=COLORS[m], linewidth=2.5,
            label=f"{MODEL_DISPLAY[m]} (AUC = {mean_auc:.4f})",
        )
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color=COLORS[m], alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves (5-seed mean ± 1 SD)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig2_roc_curves.png")


# ============================================================
# FIG 3 — PR curves
# ============================================================
def fig_pr_curves(summary):
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_recall = np.linspace(0, 1, 100)

    for m in MODELS:
        prec_per_seed = []
        ap_per_seed = []
        for s in SEEDS:
            y_test, y_proba = load_probas(m, s)
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            # Need to flip and interp on monotonically increasing recall
            order = np.argsort(rec)
            prec_sorted = prec[order]
            rec_sorted = rec[order]
            prec_interp = np.interp(mean_recall, rec_sorted, prec_sorted)
            prec_per_seed.append(prec_interp)
            ap_per_seed.append(np.trapz(prec_sorted, rec_sorted))
        prec_per_seed = np.array(prec_per_seed)
        mean_prec = prec_per_seed.mean(axis=0)
        std_prec = prec_per_seed.std(axis=0)
        r_mean = summary["per_model"][m]["auc_pr"]["mean"]
        ax.plot(
            mean_recall, mean_prec, color=COLORS[m], linewidth=2.5,
            label=f"{MODEL_DISPLAY[m]} (AUC-PR = {r_mean:.4f})",
        )
        ax.fill_between(
            mean_recall,
            np.maximum(mean_prec - std_prec, 0),
            np.minimum(mean_prec + std_prec, 1),
            color=COLORS[m], alpha=0.15,
        )

    # Baseline (class prior)
    y_test, _ = load_probas("coord", 42)
    baseline = float(np.mean(y_test))
    ax.axhline(
        baseline, color="k", linestyle="--", linewidth=1,
        label=f"Baseline ({baseline:.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (5-seed mean ± 1 SD)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig3_pr_curves.png")


# ============================================================
# FIG 4 — F1 vs threshold
# ============================================================
def fig_f1_vs_threshold(summary):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    thresholds = np.arange(0.05, 0.96, 0.02)
    for m in MODELS:
        f1_per_seed_thresh = np.zeros((len(SEEDS), len(thresholds)))
        for i, s in enumerate(SEEDS):
            y_test, y_proba = load_probas(m, s)
            for j, t in enumerate(thresholds):
                y_pred = (y_proba >= t).astype(int)
                f1_per_seed_thresh[i, j] = f1_score(
                    y_test, y_pred, zero_division=0,
                )
        mean_f1 = f1_per_seed_thresh.mean(axis=0)
        std_f1 = f1_per_seed_thresh.std(axis=0)
        ax.plot(
            thresholds, mean_f1, color=COLORS[m], linewidth=2.5,
            label=MODEL_DISPLAY[m],
        )
        ax.fill_between(
            thresholds, mean_f1 - std_f1, mean_f1 + std_f1,
            color=COLORS[m], alpha=0.15,
        )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default 0.5")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1 vs Decision Threshold (5-seed mean ± 1 SD)")
    ax.legend(loc="lower center")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_f1_vs_threshold.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ fig4_f1_vs_threshold.png")


# ============================================================
# FIG 5 — Confusion matrices side-by-side (median seed)
# ============================================================
def fig_confusion_matrices(summary):
    # Pick the seed closest to per-model median accuracy
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, MODELS):
        # Choose median-seed for this model based on accuracy
        accs = summary["per_model"][m]["accuracy"]["values"]
        median_idx = int(np.argsort(accs)[len(accs) // 2])
        seed = SEEDS[median_idx]

        y_test, y_proba = load_probas(m, seed)
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        cm_pct = cm.astype(float) / cm.sum() * 100
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticklabels(["Normal", "Abnormal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
        ax.set_title(f"{MODEL_DISPLAY[m]}\n(seed {seed}, acc = {acc:.1f}%)")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "fig5_confusion_matrices.png", dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"✓ fig5_confusion_matrices.png")


# ============================================================
# FIG 6 — Parameter efficiency (accuracy vs params)
# ============================================================
def fig_param_efficiency(summary):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for m in MODELS:
        params = summary["params"][m]
        acc_mean = summary["per_model"][m]["accuracy"]["mean"] * 100
        acc_lo = summary["per_model"][m]["accuracy"]["ci_lower"] * 100
        acc_hi = summary["per_model"][m]["accuracy"]["ci_upper"] * 100
        ax.errorbar(
            params, acc_mean,
            yerr=[[acc_mean - acc_lo], [acc_hi - acc_mean]],
            fmt="o", color=COLORS[m], markersize=14,
            capsize=6, elinewidth=2, capthick=2,
            label=f"{MODEL_DISPLAY[m]} ({params:,} params)",
        )
        ax.text(
            params, acc_mean + 0.7,
            f"{acc_mean:.2f}%", ha="center", fontsize=10,
            color=COLORS[m], fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Parameter Efficiency: Accuracy vs Model Size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right")
    ax.set_ylim(80, 92)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "fig6_param_efficiency.png", dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"✓ fig6_param_efficiency.png")


# ============================================================
# Main
# ============================================================
def main():
    summary = load_summary()
    fig_metric_bars(summary)
    fig_roc_curves(summary)
    fig_pr_curves(summary)
    fig_f1_vs_threshold(summary)
    fig_confusion_matrices(summary)
    fig_param_efficiency(summary)
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
