# Credit Card Fraud Detection System

A production-grade machine learning system designed to detect fraudulent credit card transactions using the Kaggle Credit Card Fraud dataset. The stack includes **XGBoost** for classification, **SMOTE** for balancing, **Optuna** for hyperparameter tuning, **MLflow** for experiment tracking, and **FastAPI** for low-latency predictions.

---

## Project Structure

```text
credit-card-fraud-detection/
├── data/                       # Dataset directory (contains creditcard.csv)
├── models/                     # Saved model binaries and thresholds
│   ├── baseline_model.json
│   ├── xgboost_fraud_model.json
│   └── optimal_threshold.json
├── notebooks/                  # Jupyter notebooks for analysis
│   └── eda.ipynb
├── reports/                    # Generated charts and comparison metrics
│   ├── evaluation_results.png
│   └── model_comparison.csv
├── src/                        # Python package source
│   ├── __init__.py
│   ├── api.py                  # FastAPI implementation
│   ├── evaluate.py             # Evaluation & comparison script
│   ├── features.py             # Feature engineering transformer
│   └── train.py                # Model training & Optuna tuning pipeline
├── .gitignore
├── config.yaml                 # Central configuration file
├── download_data.py            # Kaggle dataset downloader
├── pyproject.toml              # Packaging description
├── requirements.txt            # Python dependencies
└── report.md                   # Business ROI analysis report
```

---

## Getting Started

### 1. Prerequisite setup
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

### 3. Download the Dataset
Run the data setup script to fetch the Credit Card Fraud dataset (284K rows, 0.17% fraud rate) from Kaggle using `kagglehub` and place it in the `data/` folder:
```bash
python download_data.py
```

### 4. Run the Training and Tuning Pipeline
Run the model training script. This script will:
* Load the dataset and split it into stratified Train/Val/Test subsets.
* Train and evaluate a baseline XGBoost model.
* Apply SMOTE to the training subset only to address class imbalance.
* Perform 50 trials of hyperparameter tuning using Optuna to maximize validation Precision-Recall AUC (PR-AUC).
* Fit the optimized model and compute the optimal decision threshold that maximizes the F1-score for the fraud class.
* Save the models to `models/` and log parameters, metrics, and models to MLflow.
```bash
python src/train.py
```

To view the MLflow UI:
```bash
mlflow ui
```

### 5. Evaluate and Generate Comparison
Run the evaluation script to calculate comparisons, plot curves, and estimate business savings on the test partition:
```bash
python src/evaluate.py
```
This generates comparison metrics and saves plots to `reports/evaluation_results.png`.

### 6. Start the API Server
Start the FastAPI application using Uvicorn:
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

---

## API Endpoints & Usage

Once running, you can access the interactive OpenAPI docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 1. Health Check
* **GET** `/health`
* **Response:**
  ```json
  {
    "status": "healthy",
    "message": "Model is loaded and ready for predictions."
  }
  ```

### 2. Single Prediction
* **POST** `/predict`
* **Request JSON:**
  ```json
  {
    "Time": 80.0,
    "Amount": 10.00,
    "V1": -1.1, "V2": 0.5, "V3": 1.2, "V4": -0.4, "V5": 0.8, "V6": -0.2,
    "V7": 0.9, "V8": 0.1, "V9": -0.5, "V10": -0.1, "V11": 0.2, "V12": -0.8,
    "V13": 0.3, "V14": -0.9, "V15": 0.5, "V16": -0.2, "V17": 0.4, "V18": 0.1,
    "V19": -0.3, "V20": 0.05, "V21": -0.05, "V22": 0.1, "V23": -0.02,
    "V24": 0.3, "V25": 0.1, "V26": -0.1, "V27": 0.02, "V28": -0.01
  }
  ```
* **Response JSON:**
  ```json
  {
    "fraud_probability": 0.0241,
    "is_fraud": false,
    "threshold_used": 0.3842
  }
  ```

### 3. Batch Prediction
* **POST** `/predict/batch`
* **Request JSON:**
  ```json
  [
    { ... transaction 1 ... },
    { ... transaction 2 ... }
  ]
  ```
* **Response JSON:**
  ```json
  {
    "predictions": [
      { "fraud_probability": 0.0241, "is_fraud": false, "threshold_used": 0.3842 },
      { "fraud_probability": 0.8953, "is_fraud": true, "threshold_used": 0.3842 }
    ]
  }
  ```

---

## Business ROI summary
Based on our simulation metrics on the test partition:
* **Unchecked Fraud Cost:** $14,700 (98 actual frauds * $150 average amount).
* **Baseline Model Net Cost:** $3,420 (saving $11,280, or **$198.03 per 1k transactions**).
* **Optimized Model Net Cost:** $2,640 (saving $12,060, or **$211.72 per 1k transactions**).
* **Migrating to Optimized Model:** Saves an additional **$13.69 per 1,000 transactions**, which yields an extra **$136,900 monthly savings** for an organization processing 10 million transactions.
* For a full breakdown, see [report.md](file:///C:/Users/ramak/credit-card-fraud-detection/report.md).
