# 🔮 Customer Churn Prediction with Explainable AI (SHAP)

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Scikit--Learn-green.svg)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP-blueviolet.svg)](https://github.com/shap/shap)

An end-to-end Machine Learning pipeline designed to predict telecom customer churn and provide individual prediction explanations using Explainable AI (SHAP). The system features an interactive, highly visual Streamlit dashboard containing business KPIs, single customer profiling with actionable retention recommendations, batch CSV scoring, and detailed model diagnostics.

---

## 📌 Project Overview
Customer churn is a critical metric for subscription-based telecom businesses. Acquiring new customers is often significantly more expensive than retaining existing ones. 

This project goes beyond binary predictions (`Churn` or `No Churn`) by:
1. **Predicting Churn Probability**: Mapping customer risk categories (Low, Medium, High).
2. **Explaining Predictions with SHAP (Shapley Additive exPlanations)**: Outlining the specific features pushing a customer's churn risk up or down (e.g., Month-to-Month contract, Fiber Optic connection).
3. **Prescribing Actionable Recommendations**: Providing specific business interventions (e.g., payment auto-pay discounts, long-term contract bundles) based on the customer's attributes.

---

## 🚀 Key Features

*   **Executive Business Dashboard**: Tracks aggregate metrics including overall customer count, churn rates, contract distributions, and payment methods.
*   **Single Customer Profiling**: Interactive sliders and selectors to evaluate new/prospective customers in real-time.
*   **Prescriptive Retention Analytics**: Generates personalized recommendations for high-risk customers based on their features.
*   **Explainable AI (XAI)**: Renders a **SHAP Waterfall Plot** for single customer inferences to ensure transparency in model decision-making.
*   **Batch Scoring Pipeline**: Allows bulk uploading of a customer CSV, predicts churn probabilities, segments them into risk tiers, and provides a download link for the scored file.
*   **Model Diagnostics Hub**: Displays accuracy, F1-scores, ROC curves, confusion matrices, and feature importances for comparing Logistic Regression, Random Forests, and XGBoost models.

---

## 📁 Repository Directory Structure

```
Customer/
├── .venv/                      # Python virtual environment
├── data/
│   └── Telco-Customer-Churn.csv # IBM Telecom Dataset (7,043 rows, 21 columns)
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis (executed)
│   └── Model_Training.ipynb    # Model baseline prototyping (executed)
├── src/
│   ├── preprocessing.py        # Cleaning, encoding, and scaling utilities
│   ├── train.py                # Hyperparameter tuning, training, and evaluation
│   ├── predict.py              # Single/batch inference and business logic
│   └── explain.py              # SHAP waterfall plotting functions
├── models/
│   ├── churn_model.pkl         # Saved tuned XGBoost classifier
│   ├── scaler.pkl              # Saved numeric StandardScaler
│   ├── feature_cols.pkl        # Saved column names alignment file
│   └── evaluation_report.json  # Model comparison metrics JSON
├── dashboard/
│   └── app.py                  # Streamlit dashboard application
├── reports/
│   └── images/                 # Exported correlation heatmaps, curves, and confusion matrices
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Tech Stack & Libraries

*   **Language**: Python
*   **Dashboard UI**: Streamlit
*   **Data Analysis**: Pandas, NumPy
*   **Visualizations**: Plotly, Seaborn, Matplotlib
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Explainable AI**: SHAP
*   **Serialization**: Joblib

---

## 📊 Model Performance & Evaluation

Multiple models were trained on 80% of the dataset and evaluated on the remaining 20% test set. Hyperparameter tuning was performed on XGBoost using 3-fold cross-validated `GridSearchCV` optimizing for the F1-score.

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.28% | 66.10% | 52.42% | 58.47% | 0.8403 |
| **Random Forest** | 77.72% | 60.35% | 46.24% | 52.36% | 0.8155 |
| **XGBoost (Baseline)** | 76.94% | 57.79% | 47.85% | 52.35% | 0.8180 |
| **XGBoost (Tuned)** | **80.36%** | **66.55%** | **51.88%** | **58.31%** | **0.8423** |

*The best-performing model (Tuned XGBoost) is saved as `models/churn_model.pkl`.*

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-explainable-ai.git
cd customer-churn-explainable-ai
```

### 2. Set Up Virtual Environment
*   **Windows**:
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```
*   **macOS/Linux**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Application

#### A. Run the Interactive Dashboard
Launch the Streamlit app to interact with the UI:
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

#### B. Train and Tune Models
To retrain the models, execute hyperparameter search, and save updated diagnostic figures:
```bash
python src/train.py
```

---

## 💡 How it Works (Core Pipeline Stages)

### 1. Data Processing & Cleaning (`src/preprocessing.py`)
*   Converts the `TotalCharges` column to numeric and replaces blank entries with `0.0`.
*   Applies label encoding to binary columns: `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`.
*   Applies one-hot encoding to multi-category columns: `InternetService`, `PaymentMethod`, `Contract`, `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
*   Uses `StandardScaler` to normalize numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`).

### 2. Explainability Engine (`src/explain.py`)
Uses `shap.TreeExplainer` on the tuned XGBoost model to compute Shapley values for individual inferences. This breaks down the predictions and outputs a **Waterfall Plot** illustrating each feature's contribution towards or against churn.

### 3. Prescriptive Analytics (`src/predict.py`)
Generates structured guidelines:
*   **Month-to-month Contract** ➔ Recommend offering a discounted annual upgrade.
*   **Electronic Check Payment** ➔ Suggest switching to credit card or bank auto-pay.
*   **Low Tenure & High Monthly Charges** ➔ Alert support teams for onboarding check-ins.
*   **No Online Security / Tech Support** ➔ Propose bundling security tools at a discount.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
