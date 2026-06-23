import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    classification_report,
    average_precision_score,
    roc_auc_score,
)
from typing import Dict, Tuple, Any

# Add the parent directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import CreditCardFeatureEngineer

# Set Optuna logging level to warning to prevent flooding stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load raw data, engineer features, and perform stratified Train/Val/Test splits.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    raw_path = config["data"]["raw_path"]
    target_col = config["data"]["target_col"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}. Run download_data.py first.")

    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train_val (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train (80% of 80% = 64%) and val (20% of 80% = 16%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=random_state, stratify=y_train_val
    )

    print(f"Applying feature engineering to splits...")
    engineer = CreditCardFeatureEngineer()
    X_train = engineer.fit_transform(X_train)
    X_val = engineer.transform(X_val)
    X_test = engineer.transform(X_test)

    print(f"Data shapes after feature engineering:")
    print(f"  Train: {X_train.shape}, Class distribution: {dict(y_train.value_counts())}")
    print(f"  Val:   {X_val.shape}, Class distribution: {dict(y_val.value_counts())}")
    print(f"  Test:  {X_test.shape}, Class distribution: {dict(y_test.value_counts())}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_predictions(y_true: pd.Series, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute performance metrics for a given model predictions."""
    y_preds = (y_probs >= threshold).astype(int)
    
    f1 = f1_score(y_true, y_preds)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    # Calculate precision and recall manually or from report
    report = classification_report(y_true, y_preds, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    
    return {
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "accuracy": report["accuracy"]
    }


def train_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> xgb.XGBClassifier:
    """Train a baseline XGBoost model with default parameters (no tuning, no SMOTE)."""
    print("\n--- Training Baseline Model ---")
    
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    
    # Predict on test set
    probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, probs, threshold=0.5)
    
    print(f"Baseline F1 (threshold 0.5): {metrics['f1']:.4f}")
    print(f"Baseline PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Baseline ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Log baseline to MLflow
    mlflow.log_params({f"baseline_{k}": v for k, v in model.get_params().items()})
    mlflow.log_metrics({f"baseline_{k}": v for k, v in metrics.items()})
    
    return model


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the training dataset."""
    print("\n--- Applying SMOTE to Training Data ---")
    sampling_strategy = config["smote"]["sampling_strategy"]
    random_state = config["smote"]["random_state"]
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Train Shape: {X_res.shape}, Class distribution: {dict(y_res.value_counts())}")
    return X_res, y_res


def optimize_hyperparameters(
    X_train_res: pd.DataFrame, y_train_res: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Use Optuna to tune XGBoost hyperparameters on SMOTE training data, validating on Val set."""
    print("\n--- Running Hyperparameter Optimization (Optuna) ---")
    n_trials = config["optuna"]["n_trials"]
    random_state = config["optuna"]["random_state"]
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "random_state": random_state,
            "eval_metric": "logloss",
            "n_jobs": -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_res, y_train_res)
        
        # Predict on validation set (raw, imbalanced)
        probs = model.predict_proba(X_val)[:, 1]
        
        # Optimize for Precision-Recall AUC (PR-AUC) on validation set
        pr_auc = average_precision_score(y_val, probs)
        return pr_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best Trial PR-AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    return study.best_params


def find_optimal_threshold(y_true: pd.Series, y_probs: np.ndarray) -> Tuple[float, float]:
    """Find the probability threshold that maximizes F1 score for the positive class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 score for each threshold
    # Avoid division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Exclude endpoints where precision or recall might be 0
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def main():
    config = load_config()
    
    # Set MLflow experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Load and split data
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config)
        
        # Train and log baseline model
        baseline_model = train_baseline(X_train, y_train, X_test, y_test)
        
        # Apply SMOTE to training set
        X_train_res, y_train_res = apply_smote(X_train, y_train, config)
        
        # Optimize hyperparameters using Optuna on SMOTE training and raw validation data
        best_params = optimize_hyperparameters(X_train_res, y_train_res, X_val, y_val, config)
        
        # Log optimized parameters
        mlflow.log_params(best_params)
        
        # Train final model with best params on resampled training set
        print("\n--- Training Optimized Model ---")
        best_model = xgb.XGBClassifier(**best_params, random_state=config["optuna"]["random_state"])
        best_model.fit(X_train_res, y_train_res)
        
        # Evaluate optimized model on Val set to find optimal threshold
        val_probs = best_model.predict_proba(X_val)[:, 1]
        optimal_threshold, val_f1 = find_optimal_threshold(y_val, val_probs)
        print(f"Optimal Threshold (Val F1 Max): {optimal_threshold:.4f} (Val F1: {val_f1:.4f})")
        
        # Evaluate on Test set with default threshold (0.5) and optimal threshold
        test_probs = best_model.predict_proba(X_test)[:, 1]
        
        metrics_default = evaluate_predictions(y_test, test_probs, threshold=0.5)
        metrics_optimized = evaluate_predictions(y_test, test_probs, threshold=optimal_threshold)
        
        print("\n--- Test Set Evaluation ---")
        print(f"Optimized Model (th=0.5) - F1: {metrics_default['f1']:.4f}, PR-AUC: {metrics_default['pr_auc']:.4f}, ROC-AUC: {metrics_default['roc_auc']:.4f}")
        print(f"Optimized Model (th={optimal_threshold:.4f}) - F1: {metrics_optimized['f1']:.4f}, Precision: {metrics_optimized['precision']:.4f}, Recall: {metrics_optimized['recall']:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({f"opt_{k}": v for k, v in metrics_optimized.items()})
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        mlflow.log_metric("opt_f1_default_threshold", metrics_default["f1"])
        
        # Save model and threshold configs
        model_path = config["model"]["path"]
        threshold_path = config["model"]["threshold_path"]
        baseline_path = "models/baseline_model.json"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
        
        # Save baseline model
        baseline_model.save_model(baseline_path)
        print(f"Baseline model saved to {baseline_path}")
        
        # Save native XGBoost model
        best_model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Save threshold config
        threshold_data = {
            "optimal_threshold": float(optimal_threshold),
            "test_f1": float(metrics_optimized["f1"]),
            "test_precision": float(metrics_optimized["precision"]),
            "test_recall": float(metrics_optimized["recall"]),
            "test_pr_auc": float(metrics_optimized["pr_auc"]),
            "test_roc_auc": float(metrics_optimized["roc_auc"])
        }
        with open(threshold_path, "w") as f:
            json.dump(threshold_data, f, indent=4)
        print(f"Optimal threshold configuration saved to {threshold_path}")
        
        # Log XGBoost model to MLflow
        mlflow.xgboost.log_model(best_model, "model")
        
        # Log threshold configuration as artifact
        mlflow.log_artifact(threshold_path)
        
        print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
