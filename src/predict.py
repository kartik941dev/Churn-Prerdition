import os
import joblib
import pandas as pd
import numpy as np
from preprocessing import preprocess_data

# Global variables for caching model artifacts
_MODEL = None
_SCALER = None
_FEATURE_COLS = None

def load_artifacts():
    """
    Load and cache model, scaler, and feature columns list.
    """
    global _MODEL, _SCALER, _FEATURE_COLS
    
    if _MODEL is None or _SCALER is None or _FEATURE_COLS is None:
        # Use absolute paths relative to this file to prevent issues when running from other directories
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(base_dir, "models", "churn_model.pkl")
        scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
        feature_cols_path = os.path.join(base_dir, "models", "feature_cols.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_cols_path)):
            raise FileNotFoundError(f"Model artifacts not found in {os.path.join(base_dir, 'models')}. Train the model first.")
            
        _MODEL = joblib.load(model_path)
        _SCALER = joblib.load(scaler_path)
        _FEATURE_COLS = joblib.load(feature_cols_path)
        
    return _MODEL, _SCALER, _FEATURE_COLS

def predict_churn(input_df):
    """
    Given a dataframe of raw input features (single or batch),
    clean, preprocess, and predict customer churn.
    Returns: DataFrame with added prediction, probability, and risk level.
    """
    model, scaler, feature_cols = load_artifacts()
    
    # Preprocess input (passing scaler and feature_cols to align columns)
    X_processed, _, _, _ = preprocess_data(
        input_df, is_training=False, scaler=scaler, feature_cols=feature_cols
    )
    
    # Make predictions
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = model.predict(X_processed)
    
    # Calculate Risk Levels
    risk_levels = []
    for prob in probabilities:
        if prob < 0.30:
            risk_levels.append("Low")
        elif prob < 0.70:
            risk_levels.append("Medium")
        else:
            risk_levels.append("High")
            
    # Add predictions to original input df or return a new df
    results_df = input_df.copy()
    results_df['Churn_Probability'] = probabilities
    results_df['Churn_Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
    results_df['Risk_Level'] = risk_levels
    
    return results_df, X_processed

def get_business_recommendations(row):
    """
    Generate customized business recommendations based on the customer's attributes.
    """
    recommendations = []
    
    # 1. Check contract type
    contract = str(row.get('Contract', '')).lower()
    if 'month-to-month' in contract or 'month' in contract:
        recommendations.append(
            "**Upgrade to Annual Contract**: Customer is on a month-to-month plan, which increases churn risk. Offer a 10% discount on a 1-year or 2-year contract."
        )
        
    # 2. Check payment method
    payment = str(row.get('PaymentMethod', '')).lower()
    if 'electronic check' in payment:
        recommendations.append(
            "**Promote Auto-Pay**: Electronic check users have higher churn rates. Provide a one-time $5 credit to switch to automatic credit card or bank transfer payments."
        )
        
    # 3. Tech support and online security
    support = str(row.get('TechSupport', '')).lower()
    security = str(row.get('OnlineSecurity', '')).lower()
    if 'no' in support or 'no' in security:
        recommendations.append(
            "**Value Added Services Bundling**: Recommend online security and tech support services. Offer a 1-month free trial of the 'Safe & Secure' package."
        )
        
    # 4. High Monthly Charges & Low Tenure
    tenure = float(row.get('tenure', 0))
    monthly_charges = float(row.get('MonthlyCharges', 0))
    if tenure <= 6 and monthly_charges > 70:
        recommendations.append(
            "**Onboarding Care Check**: Customer is new (≤ 6 months) with high monthly charges. Initiate a proactive customer service call to ensure they are getting full value from their services."
        )
    elif monthly_charges > 90:
        recommendations.append(
            "**Plan Optimization**: Customer has high monthly charges. Offer a plan review to see if a different package or family plan could save them money while maintaining their services."
        )
        
    # Default recommendation if list is empty
    if not recommendations:
        recommendations.append(
            "**Standard Loyalty Check**: Continue standard service delivery and run an annual satisfaction survey."
        )
        
    return recommendations

if __name__ == "__main__":
    # Test batch predictions with first 5 rows of dataset
    data_path = os.path.join("data", "Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path).head(5)
        # Ensure model is trained first
        try:
            results, _ = predict_churn(df_raw)
            print("\nSample Batch Predictions:")
            print(results[['Churn_Probability', 'Churn_Prediction', 'Risk_Level']])
            
            # Print recommendations for the first user
            print("\nRecommendations for Customer 1:")
            recs = get_business_recommendations(df_raw.iloc[0])
            for r in recs:
                print("-", r)
        except Exception as e:
            print("Error making prediction:", e)
    else:
        print("Dataset not found at data/Telco-Customer-Churn.csv")
