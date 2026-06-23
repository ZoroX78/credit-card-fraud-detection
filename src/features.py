import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class CreditCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer for the Kaggle Credit Card Fraud Detection dataset.
    This class adheres to the scikit-learn Transformer interface, ensuring it can
    be integrated into machine learning pipelines and used consistently during training and serving.
    """
    def __init__(self, v_features: List[str] = None):
        self.v_features = v_features or [f"V{i}" for i in range(1, 29)]

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CreditCardFeatureEngineer":
        """Fit the transformer. This is a stateless transformer, so this is a no-op."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw transaction dataframe by creating engineered features.
        
        Args:
            X: Pandas DataFrame containing the original credit card features.
               Expected columns: 'Time', 'Amount', and 'V1' through 'V28'.
               
        Returns:
            Pandas DataFrame with both original and new engineered features.
        """
        # Create a copy to prevent modifying original data
        X_out = X.copy()

        # 1. Amount Log Transform: Amount has highly skewed distribution, log transform stabilizes variance
        X_out["Amount_Log"] = np.log1p(X_out["Amount"])

        # 2. Time transformation: Time is in seconds from first transaction.
        # 1 day = 86400 seconds. Extract hour of the day (0-23)
        X_out["Hour_Of_Day"] = (X_out["Time"] // 3600) % 24

        # 3. Statistical summary of V-features per transaction (representing overall component behavior)
        v_cols = [col for col in self.v_features if col in X_out.columns]
        if v_cols:
            X_out["V_Mean"] = X_out[v_cols].mean(axis=1)
            X_out["V_Std"] = X_out[v_cols].std(axis=1)
            # Ratios can sometimes identify outlier activities
            X_out["Amount_to_V_Ratio"] = X_out["Amount"] / (X_out[v_cols].abs().mean(axis=1) + 1e-5)
        else:
            # Fallback if V columns are not present
            X_out["V_Mean"] = 0.0
            X_out["V_Std"] = 0.0
            X_out["Amount_to_V_Ratio"] = 0.0

        # 4. Feature Interactions for top predictive components (typically V17, V14, V12, V10 in literature)
        # We will create interaction terms for V17 and V14, and V12 and V10
        if "V17" in X_out.columns and "V14" in X_out.columns:
            X_out["V17_x_V14"] = X_out["V17"] * X_out["V14"]
        else:
            X_out["V17_x_V14"] = 0.0

        if "V12" in X_out.columns and "V10" in X_out.columns:
            X_out["V12_x_V10"] = X_out["V12"] * X_out["V10"]
        else:
            X_out["V12_x_V10"] = 0.0

        return X_out

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to apply the feature engineering pipeline.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Engineered DataFrame.
    """
    engineer = CreditCardFeatureEngineer()
    return engineer.fit_transform(df)
