import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve

# =============================================================================
# CONFIGURATION
# =============================================================================
TASK4_DIR   = "task4_outputs"
OUTPUT_DIR  = "task5_outputs"
MODELS      = ["logistic_regression", "random_forest", "gbt"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "GBT"]
COLOURS     = ["#185FA5", "#3B6D11", "#854F0B"]   # blue, green, amber

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD RESULTS
# =============================================================================
def load_results():
    """Load model comparison JSON and probability CSVs from Task 4."""

    print(f"\n{'='*60}")
    print("  TASK 5 — Visualisation & Evaluation")
    print(f"{'='*60}")

    with open(f"{TASK4_DIR}/model_results.json") as f:
        results = json.load(f)

    results_df = pd.read_csv(f"{TASK4_DIR}/model_comparison.csv")
    print(f"\n  Loaded results for {len(results)} models")
    print(results_df[["model", "auc_roc", "auc_pr", "fraud_recall", "fraud_f1"]].to_string(index=False))

    prob_dfs = {}
    for name, label in zip(MODELS, MODEL_NAMES):
        path = f"{TASK4_DIR}/{name}_probs.csv"
        if os.path.exists(path):
            prob_dfs[label] = pd.read_csv(path)
            print(f"  Loaded  {path}  ({len(prob_dfs[label]):,} rows)")
        else:
            print(f"  [WARN] Missing: {path} — run Task 4 first")

    return results, results_df, prob_dfs


# =============================================================================
# 2. ROC CURVES
# =============================================================================
def plot_roc_curves(prob_dfs: dict, results_df: pd.DataFrame):
    """
    Receiver Operating Characteristic (ROC) curves for all three models.

    The ROC curve plots True Positive Rate (TPR = Recall) against
    False Positive Rate (FPR = 1 - Specificity) at all classification
    thresholds. AUC-ROC = 1.0 is perfect; AUC-ROC = 0.5 is random.

    Note: For severely imbalanced datasets (0.172% fraud), AUC-ROC can
    be misleadingly high even for poor models, because the large TN pool
    keeps FPR low. The PR curve (Figure 8) is a more informative metric.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, prob_df), colour in zip(prob_dfs.items(), COLOURS):
        y_true  = prob_df["Class"].values
        y_score = prob_df["prob_fraud"].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — Credit Card Fraud Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    axins = ax.inset_axes([0.05, 0.55, 0.40, 0.40])
    for (name, prob_df), colour in zip(prob_dfs.items(), COLOURS):
        y_true  = prob_df["Class"].values
        y_score = prob_df["prob_fraud"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axins.plot(fpr, tpr, color=colour, linewidth=1.5)
    axins.set_xlim([0.0, 0.05])
    axins.set_ylim([0.90, 1.0])
    axins.set_title("Zoom: FPR [0, 0.05]", fontsize=8)
    axins.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig07_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {path}")


# =============================================================================
# 3. PRECISION-RECALL CURVES
# =============================================================================
def plot_pr_curves(prob_dfs: dict):
    """
    Precision-Recall (PR) Curves — preferred metric for imbalanced datasets.

    For imbalanced fraud detection:
      Precision = TP / (TP + FP) — of predicted fraud, what fraction is real?
      Recall    = TP / (TP + FN) — of all real fraud, what fraction did we catch?

    The PR curve shows the precision-recall trade-off across all thresholds.
    AUC-PR is more sensitive to model quality on the minority (fraud) class
    than AUC-ROC. A random classifier achieves AUC-PR ≈ class prevalence
    (0.00172 for this dataset), so any meaningful model should far exceed this.

    High recall is operationally critical: missing a fraud (FN) imposes
    the full cost of the fraudulent transaction on the cardholder/bank.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, prob_df), colour in zip(prob_dfs.items(), COLOURS):
        y_true  = prob_df["Class"].values
        y_score = prob_df["prob_fraud"].values

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_prec             = average_precision_score(y_true, y_score)

        ax.plot(recall, precision, color=colour, linewidth=2,
                label=f"{name}  (AP = {avg_prec:.4f})")

    prevalence = prob_dfs[list(prob_dfs.keys())[0]]["Class"].mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", linewidth=1,
               label=f"Random  (AP = {prevalence:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Fraud Detection (Imbalanced)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    for (name, prob_df), colour in zip(prob_dfs.items(), COLOURS):
        y_true  = prob_df["Class"].values
        y_score = prob_df["prob_fraud"].values
        prec, rec, thresh = precision_recall_curve(y_true, y_score)
        # Find index where recall is closest to 0.9
        idx = np.argmin(np.abs(rec - 0.9))
        ax.plot(rec[idx], prec[idx], "o", color=colour, markersize=8)
        ax.annotate(
            f" r=0.9, p={prec[idx]:.2f}",
            (rec[idx], prec[idx]),
            fontsize=8, color=colour
        )

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig08_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 4. CONFUSION MATRICES
# =============================================================================
def plot_confusion_matrices(prob_dfs: dict, results_df: pd.DataFrame):
    """
    Confusion matrices for all three models at the default threshold (0.5).

    Interpreting results for fraud detection:
      True Positive  (TP): Fraud correctly detected → case investigated
      False Positive (FP): Genuine transaction blocked → customer friction
      True Negative  (TN): Genuine transaction correctly approved
      False Negative (FN): Fraud missed → financial loss (most costly)

    The confusion matrix reveals the cost-accuracy trade-off:
      High recall models increase TP and FN → fewer missed frauds
      High precision models reduce FP → fewer false blocks
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Confusion Matrices — Test Set (Default Threshold = 0.50)",
                 fontsize=14, fontweight="bold")

    for ax, (name, prob_df), colour, result_row in zip(
        axes, prob_dfs.items(), COLOURS,
        [results_df[results_df["model"]==m].iloc[0] for m in MODEL_NAMES]
    ):
        y_true = prob_df["Class"].values
        y_pred = (prob_df["prob_fraud"].values >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tp, fp, fn, tn = (
            result_row["tp"], result_row["fp"],
            result_row["fn"], result_row["tn"]
        )

        sns.heatmap(
            cm, annot=True, fmt=",d", ax=ax,
            cmap=plt.get_cmap("Blues"),
            xticklabels=["Genuine (0)", "Fraud (1)"],
            yticklabels=["Genuine (0)", "Fraud (1)"],
            linewidths=0.5, linecolor="white",
            cbar=False
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Class")
        ax.set_xlabel("Predicted Class")

        fraud_recall    = tp / (tp + fn + 1e-10)
        fraud_precision = tp / (tp + fp + 1e-10)
        ax.text(0.5, -0.18,
                f"Recall: {fraud_recall:.3f}  |  Precision: {fraud_precision:.3f}  |  "
                f"FN (missed fraud): {fn}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#A32D2D")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig09_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 5. MODEL COMPARISON BAR CHART
# =============================================================================
def plot_model_comparison(results_df: pd.DataFrame):
    """
    Side-by-side bar chart comparing all models across key metrics.
    Helps the reader immediately identify the best-performing model
    for each evaluation criterion.
    """
    metrics = {
        "AUC-ROC":         "auc_roc",
        "AUC-PR":          "auc_pr",
        "Fraud Recall":    "fraud_recall",
        "Fraud Precision": "fraud_precision",
        "Fraud F1":        "fraud_f1",
        "F2 Score":        "f2_score",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Model Comparison — Key Performance Metrics",
                 fontsize=14, fontweight="bold")

    for ax, (metric_label, col) in zip(axes.flat, metrics.items()):
        values = results_df[col].values
        bars = ax.bar(MODEL_NAMES, values, color=COLOURS, edgecolor="white",
                      linewidth=0.5, alpha=0.9)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_ylim([min(values) * 0.95, min(max(values) * 1.05, 1.0)])
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig10_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 6. FEATURE IMPORTANCE (RF & GBT side by side)
# =============================================================================
def plot_feature_importance():
    """
    Compare feature importances from Random Forest (Gini) and GBT.
    Both use impurity-based importance: weighted average reduction in
    Gini impurity / MSE across all splits on that feature.
    """
    rf_path  = f"{TASK4_DIR}/rf_feature_importances.csv"
    gbt_path = f"{TASK4_DIR}/gbt_feature_importances.csv"

    if not (os.path.exists(rf_path) and os.path.exists(gbt_path)):
        print(f"  [SKIP] Feature importance CSVs not found — run Task 4 first")
        return

    rf_fi  = pd.read_csv(rf_path).head(15)
    gbt_fi = pd.read_csv(gbt_path).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importances — Gini Impurity Reduction (Top 15)",
                 fontsize=14, fontweight="bold")

    for ax, fi_df, colour, title in zip(
        axes,
        [rf_fi, gbt_fi],
        ["#3B6D11", "#854F0B"],
        ["Random Forest", "Gradient Boosted Trees"]
    ):
        bars = ax.barh(fi_df["feature"], fi_df["importance"],
                       color=colour, alpha=0.85, edgecolor="none")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature Importance (Gini)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, fi_df["importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig11_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 7. THRESHOLD ANALYSIS (Precision-Recall trade-off at different thresholds)
# =============================================================================
def plot_threshold_analysis(prob_dfs: dict):
    """
    Shows how Precision and Recall change as the classification threshold varies.
    Enables selection of an optimal operating threshold for the business context.

    For fraud detection:
      - Low threshold (e.g. 0.3): High recall, lower precision → catch more fraud
        but generate more false alarms.
      - High threshold (e.g. 0.7): High precision, lower recall → fewer alarms
        but miss more fraud.

    The F2 score (which weights Recall twice as heavily as Precision) is
    overlaid to identify the operationally optimal threshold.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Threshold Analysis — Precision, Recall & F2 vs Classification Threshold",
                 fontsize=13, fontweight="bold")

    thresholds = np.linspace(0.01, 0.99, 200)

    for ax, (name, prob_df), colour in zip(axes, prob_dfs.items(), COLOURS):
        y_true  = prob_df["Class"].values
        y_score = prob_df["prob_fraud"].values

        precisions, recalls, f2_scores = [], [], []

        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            p  = tp / (tp + fp + 1e-10)
            r  = tp / (tp + fn + 1e-10)
            f2 = 5 * p * r / (4 * p + r + 1e-10)

            precisions.append(p)
            recalls.append(r)
            f2_scores.append(f2)

        ax.plot(thresholds, recalls,    color="#A32D2D",  label="Recall",     linewidth=2)
        ax.plot(thresholds, precisions, color="#185FA5",  label="Precision",  linewidth=2)
        ax.plot(thresholds, f2_scores,  color="#3B6D11",  label="F2 Score",   linewidth=2, linestyle="--")

        best_t = thresholds[np.argmax(f2_scores)]
        best_f2 = max(f2_scores)
        ax.axvline(x=best_t, color="#854F0B", linestyle=":", linewidth=1.5,
                   label=f"Opt. threshold={best_t:.2f}")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("Score")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        print(f"  {name}: Optimal threshold (F2) = {best_t:.3f}  "
              f"|  F2 = {best_f2:.4f}")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig12_threshold_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 8. SUMMARY TABLE (printed + saved as CSV)
# =============================================================================
def print_final_summary(results_df: pd.DataFrame):
    """
    Print a comprehensive evaluation summary and recommendation.
    """
    print(f"\n{'='*80}")
    print("  TASK 5 — FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    display_cols = [
        "model", "auc_roc", "auc_pr",
        "fraud_precision", "fraud_recall", "fraud_f1",
        "f2_score", "tp", "fn", "training_time_s"
    ]
    print(results_df[display_cols].to_string(index=False))

    best_recall_row = results_df.loc[results_df["fraud_recall"].idxmax()]
    best_auc_row    = results_df.loc[results_df["auc_roc"].idxmax()]
    best_f2_row     = results_df.loc[results_df["f2_score"].idxmax()]

    print(f"\n  Best AUC-ROC    : {best_auc_row['model']}  ({best_auc_row['auc_roc']:.4f})")
    print(f"  Best Recall     : {best_recall_row['model']}  ({best_recall_row['fraud_recall']:.4f})")
    print(f"  Best F2 Score   : {best_f2_row['model']}  ({best_f2_row['f2_score']:.4f})")

    print(f"""
  RECOMMENDATION:
  ─────────────────────────────────────────────────────────────────
  For production fraud detection, {best_recall_row['model']} is recommended
  as the primary model. Although Logistic Regression provides the fastest
  inference and highest interpretability, the non-linear ensemble models
  (Random Forest, GBT) capture complex fraud patterns that exceed the
  capacity of a linear boundary.

  The severe class imbalance (0.172% fraud) makes Recall the primary
  operational metric — every False Negative represents an undetected fraud
  transaction imposing financial loss on the cardholder or the bank.
  F2 Score (which weights Recall twice as heavily as Precision) is
  recommended as the production monitoring KPI.

  SMOTE oversampling (Task 2) significantly improved Recall over training
  on the raw imbalanced data, particularly for Logistic Regression.
  ─────────────────────────────────────────────────────────────────
""")

    results_df.to_csv(f"{OUTPUT_DIR}/final_evaluation_summary.csv", index=False)
    print(f"  [Saved] {OUTPUT_DIR}/final_evaluation_summary.csv")
    print(f"  All figures saved to: {OUTPUT_DIR}/\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    results, results_df, prob_dfs = load_results()

    if not prob_dfs:
        print("\n  ERROR: No probability CSV files found.")
        print("  Please run task4_ml_models.py first.\n")
        return

    print(f"\n  Generating visualisations...")

    plot_roc_curves(prob_dfs, results_df)
    plot_pr_curves(prob_dfs)
    plot_confusion_matrices(prob_dfs, results_df)
    plot_model_comparison(results_df)
    plot_feature_importance()
    plot_threshold_analysis(prob_dfs)
    print_final_summary(results_df)


if __name__ == "__main__":
    main()
