import os
import sys
import json
import yaml
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import CreditCardFeatureEngineer

# Global state for model and threshold
model = None
optimal_threshold = 0.5
config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load the model and configuration on startup."""
    global model, optimal_threshold, config
    
    # 1. Load config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"CRITICAL: Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Load model
    model_path = config["model"]["path"]
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}. API running in degraded state.")
    else:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"Successfully loaded XGBoost model from {model_path}")
        
    # 3. Load threshold
    threshold_path = config["model"]["threshold_path"]
    if not os.path.exists(threshold_path):
        print(f"WARNING: Threshold file not found at {threshold_path}. Using default 0.5.")
        optimal_threshold = 0.5
    else:
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get("optimal_threshold", 0.5)
        print(f"Successfully loaded optimal threshold: {optimal_threshold}")
        
    yield
    # Clean up on shutdown if needed
    print("Shutting down API server...")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-grade API serving XGBoost fraud detection model predictions.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Pydantic Schemas ---
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction in the dataset.", example=0.0)
    Amount: float = Field(..., description="Transaction amount.", example=149.62)
    V1: float = Field(..., description="PCA reduced component V1", example=-1.359807)
    V2: float = Field(..., description="PCA reduced component V2", example=-0.072781)
    V3: float = Field(..., description="PCA reduced component V3", example=2.536347)
    V4: float = Field(..., description="PCA reduced component V4", example=1.378155)
    V5: float = Field(..., description="PCA reduced component V5", example=-0.338321)
    V6: float = Field(..., description="PCA reduced component V6", example=0.462388)
    V7: float = Field(..., description="PCA reduced component V7", example=0.239599)
    V8: float = Field(..., description="PCA reduced component V8", example=0.098698)
    V9: float = Field(..., description="PCA reduced component V9", example=0.363787)
    V10: float = Field(..., description="PCA reduced component V10", example=0.090794)
    V11: float = Field(..., description="PCA reduced component V11", example=-0.551600)
    V12: float = Field(..., description="PCA reduced component V12", example=-0.617801)
    V13: float = Field(..., description="PCA reduced component V13", example=-0.991390)
    V14: float = Field(..., description="PCA reduced component V14", example=-0.311169)
    V15: float = Field(..., description="PCA reduced component V15", example=1.468177)
    V16: float = Field(..., description="PCA reduced component V16", example=-0.470401)
    V17: float = Field(..., description="PCA reduced component V17", example=0.207971)
    V18: float = Field(..., description="PCA reduced component V18", example=0.025791)
    V19: float = Field(..., description="PCA reduced component V19", example=0.403993)
    V20: float = Field(..., description="PCA reduced component V20", example=0.251412)
    V21: float = Field(..., description="PCA reduced component V21", example=-0.018307)
    V22: float = Field(..., description="PCA reduced component V22", example=0.277838)
    V23: float = Field(..., description="PCA reduced component V23", example=-0.110474)
    V24: float = Field(..., description="PCA reduced component V24", example=0.066928)
    V25: float = Field(..., description="PCA reduced component V25", example=0.128539)
    V26: float = Field(..., description="PCA reduced component V26", example=-0.189115)
    V27: float = Field(..., description="PCA reduced component V27", example=0.133558)
    V28: float = Field(..., description="PCA reduced component V28", example=-0.021053)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 80.0,
                "Amount": 10.00,
                "V1": -1.1, "V2": 0.5, "V3": 1.2, "V4": -0.4, "V5": 0.8,
                "V6": -0.2, "V7": 0.9, "V8": 0.1, "V9": -0.5, "V10": -0.1,
                "V11": 0.2, "V12": -0.8, "V13": 0.3, "V14": -0.9, "V15": 0.5,
                "V16": -0.2, "V17": 0.4, "V18": 0.1, "V19": -0.3, "V20": 0.05,
                "V21": -0.05, "V22": 0.1, "V23": -0.02, "V24": 0.3, "V25": 0.1,
                "V26": -0.1, "V27": 0.02, "V28": -0.01
            }
        }
    }

class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model calculated probability of fraud (0.0 to 1.0).")
    is_fraud: bool = Field(..., description="Binary classification outcome at the optimal threshold.")
    threshold_used: float = Field(..., description="The decision threshold used.")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# --- Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Check API status and model load status."""
    if model is None:
        return {
            "status": "degraded",
            "message": "Model not loaded. Run training script to generate the model file."
        }
    return {
        "status": "healthy",
        "message": "Model is loaded and ready for predictions."
    }


def predict_df(df_input: pd.DataFrame) -> List[Dict[str, Any]]:
    """Helper to run the inference pipeline on a DataFrame."""
    # 1. Feature Engineering
    engineer = CreditCardFeatureEngineer()
    df_engineered = engineer.transform(df_input)
    
    # 2. Run prediction
    # XGBoost needs features in the exact same order as training
    # The feature engineer preserves the base features and appends engineered ones in order
    probs = model.predict_proba(df_engineered)[:, 1]
    
    # 3. Apply threshold
    results = []
    for prob in probs:
        results.append({
            "fraud_probability": float(prob),
            "is_fraud": bool(prob >= optimal_threshold),
            "threshold_used": float(optimal_threshold)
        })
    return results


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_single(transaction: TransactionInput) -> Dict[str, Any]:
    """
    Predict fraud probability for a single transaction.
    
    Returns fraud probability, is_fraud boolean indicator, and optimal threshold used.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on this server."
        )
        
    try:
        # Convert input Pydantic model to a 1-row Pandas DataFrame
        df_input = pd.DataFrame([transaction.model_dump()])
        results = predict_df(df_input)
        return results[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(transactions: List[TransactionInput]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Predict fraud probability for a batch of transactions.
    
    Returns a list of prediction outcomes.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_53_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on this server."
        )
        
    try:
        # Convert list of Pydantic models to a multi-row Pandas DataFrame
        data = [t.model_dump() for t in transactions]
        df_input = pd.DataFrame(data)
        results = predict_df(df_input)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
