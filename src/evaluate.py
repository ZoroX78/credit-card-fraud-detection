import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import mlflow
from sklearn.metrics import (
    classification_report,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from typing import Dict, Any

# Add the parent directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import prepare_data, load_config, evaluate_predictions


def load_models(config: Dict[str, Any]) -> tuple:
    """Load the trained baseline and optimized models."""
    baseline_path = "models/baseline_model.json"
    optimized_path = config["model"]["path"]
    threshold_path = config["model"]["threshold_path"]

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model not found at {baseline_path}. Run train.py first.")
    if not os.path.exists(optimized_path):
        raise FileNotFoundError(f"Optimized model not found at {optimized_path}. Run train.py first.")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Optimal threshold file not found at {threshold_path}. Run train.py first.")

    baseline = xgb.XGBClassifier()
    baseline.load_model(baseline_path)

    optimized = xgb.XGBClassifier()
    optimized.load_model(optimized_path)

    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
    optimal_threshold = threshold_data["optimal_threshold"]

    return baseline, optimized, optimal_threshold


def plot_curves(
    y_test: pd.Series,
    baseline_probs: np.ndarray,
    optimized_probs: np.ndarray,
    optimal_threshold: float,
    save_path: str
):
    """Generate and save evaluation plots (ROC, PR, Confusion Matrix)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 1. ROC Curve ---
    fpr_b, tpr_b, _ = roc_curve(y_test, baseline_probs)
    fpr_o, tpr_o, _ = roc_curve(y_test, optimized_probs)
    roc_auc_b = roc_auc_score(y_test, baseline_probs)
    roc_auc_o = roc_auc_score(y_test, optimized_probs)

    axes[0].plot(fpr_b, tpr_b, label=f"Baseline (AUC = {roc_auc_b:.4f})", linestyle="--", color="gray")
    axes[0].plot(fpr_o, tpr_o, label=f"Optimized (AUC = {roc_auc_o:.4f})", color="royalblue", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves Comparison")
    axes[0].legend(loc="lower right")

    # --- 2. Precision-Recall Curve ---
    prec_b, rec_b, _ = precision_recall_curve(y_test, baseline_probs)
    prec_o, rec_o, thresh_o = precision_recall_curve(y_test, optimized_probs)
    pr_auc_b = average_precision_score(y_test, baseline_probs)
    pr_auc_o = average_precision_score(y_test, optimized_probs)

    axes[1].plot(rec_b, prec_b, label=f"Baseline (AP = {pr_auc_b:.4f})", linestyle="--", color="gray")
    axes[1].plot(rec_o, prec_o, label=f"Optimized (AP = {pr_auc_o:.4f})", color="forestgreen", linewidth=2)
    
    # Highlight optimal threshold on optimized PR curve
    # Find the precision/recall closest to the optimal threshold
    idx = np.argmin(np.abs(thresh_o - optimal_threshold))
    opt_rec = rec_o[idx]
    opt_prec = prec_o[idx]
    axes[1].plot(opt_rec, opt_prec, "ro", markersize=8, label=f"Optimal Th = {optimal_threshold:.4f}")
    
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves Comparison")
    axes[1].legend(loc="lower left")

    # --- 3. Confusion Matrix at Optimal Threshold ---
    opt_preds = (optimized_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, opt_preds)
    
    # Formatted labels
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=axes[2],
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_xticklabels(["Legit", "Fraud"])
    axes[2].set_yticklabels(["Legit", "Fraud"])
    axes[2].set_title(f"Optimized Model CM (Threshold = {optimal_threshold:.4f})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Evaluation plots saved to {save_path}")


def main():
    config = load_config()
    
    # Set MLflow experiment to log evaluation results under the current/latest experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load data and models
    _, _, _, _, X_test, y_test = prepare_data(config)
    baseline, optimized, optimal_threshold = load_models(config)

    # Get predictions
    baseline_probs = baseline.predict_proba(X_test)[:, 1]
    optimized_probs = optimized.predict_proba(X_test)[:, 1]

    # Evaluate configurations
    metrics_baseline = evaluate_predictions(y_test, baseline_probs, threshold=0.5)
    metrics_opt_default = evaluate_predictions(y_test, optimized_probs, threshold=0.5)
    metrics_opt_tuned = evaluate_predictions(y_test, optimized_probs, threshold=optimal_threshold)

    # Generate comparison table
    df_compare = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "ROC-AUC", "PR-AUC"],
        "Baseline (th=0.5)": [
            metrics_baseline["accuracy"],
            metrics_baseline["precision"],
            metrics_baseline["recall"],
            metrics_baseline["f1"],
            metrics_baseline["roc_auc"],
            metrics_baseline["pr_auc"]
        ],
        "Optimized (th=0.5)": [
            metrics_opt_default["accuracy"],
            metrics_opt_default["precision"],
            metrics_opt_default["recall"],
            metrics_opt_default["f1"],
            metrics_opt_default["roc_auc"],
            metrics_opt_default["pr_auc"]
        ],
        "Optimized (th=Tuned)": [
            metrics_opt_tuned["accuracy"],
            metrics_opt_tuned["precision"],
            metrics_opt_tuned["recall"],
            metrics_opt_tuned["f1"],
            metrics_opt_tuned["roc_auc"],
            metrics_opt_tuned["pr_auc"]
        ]
    })

    print("\n==================== MODEL COMPARISON TABLE ====================")
    print(df_compare.to_markdown(index=False))
    print("================================================================\n")

    # Generate plots
    plot_path = "reports/evaluation_results.png"
    plot_curves(y_test, baseline_probs, optimized_probs, optimal_threshold, plot_path)

    # Business Savings calculations
    cm_opt = confusion_matrix(y_test, (optimized_probs >= optimal_threshold).astype(int))
    tn, fp, fn, tp = cm_opt.ravel()
    
    avg_fraud = config["business"]["avg_fraud_amount"]
    cost_fp = config["business"]["cost_of_false_positive"]

    # Savings = (TP * avg_fraud) - (FP * cost_fp)
    total_savings = (tp * avg_fraud) - (fp * cost_fp)
    num_transactions = len(y_test)
    savings_per_1000 = (total_savings / num_transactions) * 1000

    print("==================== BUSINESS ROI REPORT ====================")
    print(f"Total Transactions in Test Set: {num_transactions:,}")
    print(f"True Positives (Fraud Prevented): {tp} (out of {tp + fn} total frauds)")
    print(f"False Positives (Customer Friction): {fp}")
    print(f"Assumed Avg Fraud Amount: ${avg_fraud:.2f}")
    print(f"Assumed Cost of False Positive: ${cost_fp:.2f}")
    print(f"Estimated Net Savings on Test Set: ${total_savings:,.2f}")
    print(f"Estimated Savings per 1,000 Transactions: ${savings_per_1000:.2f}")
    print("=============================================================\n")

    # Log to active MLflow run if any, otherwise log to a new run
    with mlflow.start_run(run_name="Evaluation_Report") as run:
        mlflow.log_metrics({f"eval_baseline_{k}": v for k, v in metrics_baseline.items()})
        mlflow.log_metrics({f"eval_opt_tuned_{k}": v for k, v in metrics_opt_tuned.items()})
        mlflow.log_metric("eval_savings_per_1000", savings_per_1000)
        mlflow.log_artifact(plot_path)
        
        # Save comparison table as CSV and log
        csv_path = "reports/model_comparison.csv"
        df_compare.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        print("Logged comparison CSV and plots to MLflow.")

if __name__ == "__main__":
    main()
