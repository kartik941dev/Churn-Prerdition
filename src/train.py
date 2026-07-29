import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_data

def train_models():
    # 1. Load data
    data_path = os.path.join("data", "Telco-Customer-Churn.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download command first.")
        
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully. Shape:", df.shape)
    
    # 2. Preprocess data
    X, y, scaler, feature_cols = preprocess_data(df, is_training=True)
    
    # Save the scaler and feature columns for prediction time
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join("models", "feature_cols.pkl"))
    print("Saved scaler.pkl and feature_cols.pkl to models/")
    
    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 4. Train baseline models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost (Baseline)": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        print(f"{name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
    # 5. Hyperparameter Tuning for XGBoost
    print("\n--- Tuning XGBoost Hyperparameters ---")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_base = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    print("Best XGBoost Hyperparameters found:", grid_search.best_params_)
    
    # Evaluate best model
    best_preds = best_xgb.predict(X_test)
    best_probs = best_xgb.predict_proba(X_test)[:, 1]
    
    best_acc = accuracy_score(y_test, best_preds)
    best_prec = precision_score(y_test, best_preds)
    best_rec = recall_score(y_test, best_preds)
    best_f1 = f1_score(y_test, best_preds)
    best_auc = roc_auc_score(y_test, best_probs)
    
    results["XGBoost (Tuned)"] = {
        "Accuracy": best_acc,
        "Precision": best_prec,
        "Recall": best_rec,
        "F1-Score": best_f1,
        "ROC-AUC": best_auc
    }
    
    print(f"\nTuned XGBoost Results - Acc: {best_acc:.4f}, Prec: {best_prec:.4f}, Rec: {best_rec:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    
    # Save the best model
    joblib.dump(best_xgb, os.path.join("models", "churn_model.pkl"))
    print("Saved churn_model.pkl to models/")
    
    # Save evaluation report as JSON
    with open(os.path.join("models", "evaluation_report.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("Saved evaluation_report.json to models/")
    
    # 6. Plot & Save Confusion Matrix & ROC Curve of the best model
    os.makedirs(os.path.join("reports", "images"), exist_ok=True)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Best XGBoost Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "images", "confusion_matrix.png"), dpi=300)
    plt.close()
    print("Saved confusion_matrix.png to reports/images/")
    
    # ROC Curve Plot
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "images", "roc_curve.png"), dpi=300)
    plt.close()
    print("Saved roc_curve.png to reports/images/")
    
    # Feature Importance Plot
    importance = best_xgb.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Importances - Tuned XGBoost')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "images", "feature_importance.png"), dpi=300)
    plt.close()
    print("Saved feature_importance.png to reports/images/")

if __name__ == "__main__":
    train_models()
