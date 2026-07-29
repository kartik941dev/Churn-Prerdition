import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def get_shap_explainer(model):
    """
    Initialize and return a SHAP TreeExplainer for the trained XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    return explainer

def get_shap_explanation_object(explainer, processed_row, feature_names):
    """
    Compute SHAP values and return a shap.Explanation object for a single customer row.
    """
    # Compute SHAP values for the processed row
    shap_values = explainer(processed_row)
    
    # Extract values for the current row
    # In some shap versions, explainer(X) returns an Explanation object directly.
    # If the output is a 3D or 2D array, we handle shape adjustment.
    values = shap_values.values[0]
    base_value = shap_values.base_values[0]
    data = processed_row.iloc[0].values
    
    # Create a clean shap.Explanation object which is required for shap.plots.waterfall
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data,
        feature_names=feature_names
    )
    
    return explanation

def generate_waterfall_plot(explanation, save_path, max_display=10):
    """
    Generate and save a SHAP waterfall plot for a single customer prediction.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot waterfall
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    
    # Adjust layout to fit labels
    plt.gcf().patch.set_facecolor('white')
    plt.tight_layout()
    
    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

if __name__ == "__main__":
    # Test block
    model_path = os.path.join("models", "churn_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    features_path = os.path.join("models", "feature_cols.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        
        # Create a mock processed data row (all zeros)
        mock_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
        
        explainer = get_shap_explainer(model)
        explanation = get_shap_explanation_object(explainer, mock_data, feature_cols)
        
        test_plot_path = os.path.join("reports", "images", "test_waterfall.png")
        generate_waterfall_plot(explanation, test_plot_path)
        print(f"Test SHAP waterfall plot saved to {test_plot_path}")
    else:
        print("Model files not found. Run train.py first.")
