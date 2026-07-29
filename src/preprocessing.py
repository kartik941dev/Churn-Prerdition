import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def clean_data(df):
    """
    Clean the dataset by converting datatypes, removing duplicates,
    and handling missing values.
    """
    df = df.copy()
    
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Convert TotalCharges to numeric
    # Errors = 'coerce' replaces spaces or invalid entries with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # For rows where tenure is 0, TotalCharges is typically NaN or blank. Fill with 0.
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # SeniorCitizen is binary 0/1, we can leave it as integer or map it. It is already 0/1.
    return df

def preprocess_data(df, is_training=True, scaler=None, feature_cols=None):
    """
    Encode categorical features and scale numeric features.
    If is_training is True, fits and returns a new scaler and feature list.
    If is_training is False, uses the provided scaler and aligns columns with feature_cols.
    """
    df = clean_data(df)
    
    # Binary columns to label-encode (0/1 mapping)
    binary_cols = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
            
    # Churn is the target (if training)
    y = None
    if 'Churn' in df.columns:
        y = df['Churn'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        df = df.drop(columns=['Churn'])
        
    # Columns to one-hot encode
    # We include all remaining categorical columns to keep their predictive power
    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ]
    
    # Keep only columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Ensure boolean columns from get_dummies are converted to 0/1 int
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
            
    # Numerical columns to scale
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numeric_cols = [col for col in numeric_cols if col in df_encoded.columns]
    
    if is_training:
        scaler = StandardScaler()
        if len(numeric_cols) > 0:
            df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
            
        feature_cols = list(df_encoded.columns)
        return df_encoded, y, scaler, feature_cols
    else:
        # If testing/predicting, align columns with training features
        # Add missing columns with 0
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder to match training features and drop any extra columns
        df_encoded = df_encoded[feature_cols]
        
        # Scale numeric columns
        if len(numeric_cols) > 0 and scaler is not None:
            df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
            
        return df_encoded, None, scaler, feature_cols

if __name__ == "__main__":
    # Test block
    data_path = os.path.join("data", "Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        print("Raw shape:", df_raw.shape)
        X, y, scaler, feature_cols = preprocess_data(df_raw, is_training=True)
        print("Processed shape:", X.shape)
        print("Target sum (churn count):", y.sum())
        print("First 3 processed features:\n", X.iloc[:3, :5])
    else:
        print("Dataset not found at data/Telco-Customer-Churn.csv")
