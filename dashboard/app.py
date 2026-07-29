import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_churn, get_business_recommendations, load_artifacts
from explain import get_shap_explainer, get_shap_explanation_object, generate_waterfall_plot

# Set page config for a premium look
st.set_page_config(
    page_title="Customer Churn Insights & Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for premium look
st.markdown("""
<style>
    /* Theme color variables and overrides */
    .stApp {
        background-color: #374151  ;
        color: #F9FAFB;
    }
    .main {
        background-color: #000000 ;
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #475569  !important;
        font-weight: 600;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1.5px solid #FCA5A5;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.05), 0 2px 4px -1px rgba(220, 38, 38, 0.03);
    }
    .recommendation-box {
        background-color: #EEF2FF;
        border-left: 5px solid #6366F1;
        border-top: 1px solid #E0E7FF;
        border-right: 1px solid #E0E7FF;
        border-bottom: 1px solid #E0E7FF;
        padding: 15px;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 10px;
        color: #312E81;
    }
    button[data-baseweb="tab"] p {
        color: #64748B !important;
        font-weight: 600;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #7F1D1D !important;
    }
    .badge-high {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-medium {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# App Title with custom modern design
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; background-color: #FEE2E2; padding: 20px; border-radius: 16px; border: 1.5px solid #FCA5A5;">
    <h1 style="color: #991B1B; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 800;">🔮 Customer Churn Analytics Dashboard</h1>
    <p style="color: #7F1D1D; font-size: 1.1rem; font-weight: 500; margin: 0;">Predict customer churn, explain factors with SHAP, and get actionable retention recommendations.</p>
</div>
""", unsafe_allow_html=True)

# Helper function to check if models exist
def check_models_exist():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl'))
    return os.path.exists(model_path)

if not check_models_exist():
    st.warning("⚠️ Churn prediction model has not been trained yet. Please run the model training script first.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model, tuning hyperparameters, and computing SHAP explainer..."):
            try:
                # Import train script here to avoid issues if imports aren't fully set up yet
                import subprocess
                train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'))
                result = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("🎉 Model trained successfully! Refreshing dashboard...")
                    st.rerun()
                else:
                    st.error(f"Error during training: {result.stderr}")
            except Exception as e:
                st.error(f"Failed to execute training script: {e}")
    st.stop()

# Load Dataset for overview charts
@st.cache_data
def load_raw_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv'))
    return pd.read_csv(data_path)

try:
    df_raw = load_raw_data()
except Exception as e:
    st.error(f"Could not load data/Telco-Customer-Churn.csv. Ensure the dataset is downloaded. Error: {e}")
    st.stop()

# Load Model components
try:
    model, scaler, feature_cols = load_artifacts()
    explainer = get_shap_explainer(model)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Tab setup-
tab_overview, tab_single, tab_batch, tab_performance = st.tabs([
    "📈 Overview Dashboard", 
    "👤 Single Customer Prediction", 
    "📂 Batch Predictions (CSV)", 
    "⚙️ Model Insights & Metrics"
])

# ==================== OVERVIEW DASHBOARD ====================
with tab_overview:
    st.subheader("Business Metrics Overview")
    
    # Calculate metrics
    total_customers = df_raw.shape[0]
    churn_rate = (df_raw['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    avg_tenure = df_raw['tenure'].mean()
    
    # Clean TotalCharges for calculating stats
    total_charges_numeric = pd.to_numeric(df_raw['TotalCharges'], errors='coerce').fillna(0)
    avg_monthly_charges = df_raw['MonthlyCharges'].mean()
    
    # Render premium KPI metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Customers</h3>
            <h2 style="color: #38BDF8; font-size: 2.2rem; margin: 0;">{total_customers:,}</h2>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 5px 0 0 0;">Active in Database</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Churn Rate</h3>
            <h2 style="color: #EF4444; font-size: 2.2rem; margin: 0;">{churn_rate:.1f}%</h2>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 5px 0 0 0;">Avg telecom churn ~20%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Tenure</h3>
            <h2 style="color: #10B981; font-size: 2.2rem; margin: 0;">{avg_tenure:.1f} mo</h2>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 5px 0 0 0;">Customer lifetime length</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Monthly Bill</h3>
            <h2 style="color: #F59E0B; font-size: 2.2rem; margin: 0;">${avg_monthly_charges:.2f}</h2>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 5px 0 0 0;">Per customer revenue</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row of charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### Churn Distribution")
        fig_pie = px.pie(
            df_raw, names='Churn', 
            hole=0.4, 
            color_discrete_sequence=["#7CAA11", '#EF4444'],
            labels={'Churn': 'Has Churned?'}
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#1E293B',
            legend=dict(orientation="h", y=1.1, x=0.3)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        st.markdown("### Tenure vs. Churn")
        fig_hist = px.histogram(
            df_raw, x='tenure', color='Churn',
            barmode='overlay',
            color_discrete_sequence=["#0ADD97", "#E72121"],
            nbins=30
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#1E293B',
            xaxis_title="Tenure (Months)",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.1, x=0.3)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown("### Contract Type vs. Churn")
        fig_contract = px.histogram(
            df_raw, x='Contract', color='Churn',
            barmode='group',
            color_discrete_sequence=['#10B981', '#EF4444']
        )
        fig_contract.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#274B85",
            xaxis_title="Contract Type",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.1, x=0.3)
        )
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with chart_col4:
        st.markdown("### Payment Method vs. Churn")
        fig_payment = px.histogram(
            df_raw, x='PaymentMethod', color='Churn',
            barmode='group',
            color_discrete_sequence=['#10B981', '#EF4444']
        )
        fig_payment.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#1E293B',
            xaxis_title="Payment Method",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.1, x=0.3)
        )
        st.plotly_chart(fig_payment, use_container_width=True)

# ==================== SINGLE CUSTOMER PREDICTION ====================
with tab_single:
    st.subheader("Predict and Explain Single Customer Churn")
    
    # Input panels split into columns
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner (Married/Cohabiting)", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months with Company)", 0, 72, 12)

    with col_input2:
        st.markdown("### Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service" if phone_service == "No" else "No", "Yes"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        
        # Online security etc are only relevant if Internet Service is not No
        no_internet_opt = "No internet service" if internet_service == "No" else "No"
        
        online_security = st.selectbox("Online Security", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", [no_internet_opt, "Yes"] if internet_service == "No" else ["No", "Yes"])

    with col_input3:
        st.markdown("### Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=1.0)
        
        # Pre-fill Total Charges with a logical default (tenure * monthly_charges)
        default_total = float(tenure * monthly_charges)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=default_total, step=10.0)

    # Compile input data into a single-row DataFrame
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([input_data])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Run Prediction
    if st.button("🔮 Analyze & Predict Churn Risk"):
        with st.spinner("Analyzing customer risk profile..."):
            results_df, processed_row = predict_churn(input_df)
            
            prob = results_df.iloc[0]['Churn_Probability']
            prediction = results_df.iloc[0]['Churn_Prediction']
            risk = results_df.iloc[0]['Risk_Level']
            
            # Displays columns for results
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("### Prediction Result")
                
                # Custom risk badge
                if risk == "High":
                    badge_html = f'<span class="badge-high">HIGH RISK ({prob:.1%})</span>'
                elif risk == "Medium":
                    badge_html = f'<span class="badge-medium">MEDIUM RISK ({prob:.1%})</span>'
                else:
                    badge_html = f'<span class="badge-low">LOW RISK ({prob:.1%})</span>'
                    
                st.markdown(f"**Status**: {badge_html}", unsafe_allow_html=True)
                st.markdown(f"**Will Churn?**: **{'Yes' if prediction == 'Yes' else 'No'}**")
                
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability %", 'font': {'size': 18, 'color': "#7F1D1D"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E293B"},
                        'bar': {'color': "#EF4444"},
                        'bgcolor': "#335272",
                        'borderwidth': 1.5,
                        'bordercolor': "#813232",
                        'steps': [
                            {'range': [0, 30], 'color': "#07703F"},
                            {'range': [30, 70], 'color': "#A58F37"},
                            {'range': [70, 100], 'color': '#FCA5A5'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#1E293B", 'family': "Inter"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res2:
                st.markdown("### Actionable Recommendations")
                recommendations = get_business_recommendations(input_data)
                
                # If high risk, add a critical note
                if risk == "High":
                    st.markdown("> 🚨 **Critical Customer Alert**: Churn risk is extremely high. Offer direct incentives immediately.")
                    
                for rec in recommendations:
                    st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

            # Generate and render SHAP Plot
            st.markdown("<br>### 🔍 Why did this prediction happen? (SHAP Explanations)", unsafe_allow_html=True)
            try:
                explanation = get_shap_explanation_object(explainer, processed_row, feature_cols)
                
                # Save plot to temp image
                temp_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'temp_waterfall.png'))
                generate_waterfall_plot(explanation, temp_plot_path)
                
                # Show the image
                st.image(temp_plot_path, caption="SHAP Waterfall Plot: Red bars push the probability up, blue bars pull it down.")
            except Exception as e:
                st.error(f"Could not render SHAP Waterfall plot: {e}")

# ==================== BATCH PREDICTIONS ====================
with tab_batch:
    st.subheader("Batch Customer Prediction")
    st.write("Upload a CSV file containing customer records to score them in bulk. The CSV should contain the same feature columns as the raw Telco dataset.")
    
    # CSV Template Downloader
    st.markdown("#### Sample Input Format")
    sample_df = df_raw.head(3).drop(columns=['Churn'])
    st.dataframe(sample_df)
    
    uploaded_file = st.file_uploader("Upload Customer CSV File", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file containing {batch_raw.shape[0]} records.")
            
            # Predict
            with st.spinner("Scoring batch records..."):
                scored_df, _ = predict_churn(batch_raw)
                
                # Show preview of scored dataset
                st.markdown("### Predicted Results Preview")
                scored_preview = scored_df[['gender', 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability', 'Churn_Prediction', 'Risk_Level']]
                st.dataframe(scored_preview.head(20))
                
                # Download link for scored file
                csv_data = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored CSV File",
                    data=csv_data,
                    file_name="churn_predictions_scored.csv",
                    mime="text/csv"
                )
                
                # Show aggregate predictions count
                st.markdown("### Prediction Summary Statistics")
                fig_bar = px.histogram(scored_df, x='Risk_Level', color='Churn_Prediction', barmode='group',
                                     color_discrete_sequence=['#10B981', '#EF4444'])
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#1E293B',
                    legend=dict(orientation="h", y=1.1, x=0.3)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ==================== MODEL PERFORMANCE INSIGHTS ====================
with tab_performance:
    st.subheader("Model Performance Evaluation")
    
    # Load evaluation report
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'evaluation_report.json'))
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            eval_report = json.load(f)
            
        st.markdown("### Model Comparison Metrics")
        
        # Format into a table
        metrics_df = pd.DataFrame(eval_report).T
        st.table(metrics_df.style.highlight_max(axis=0, color='#1E293B'))
        
        # Explain Tuned XGBoost
        st.markdown("""
        The final selected model is **Tuned XGBoost** due to its high F1-Score and ROC-AUC. 
        It has been saved to `models/churn_model.pkl`.
        """)
        
        # Display Saved Images
        st.markdown("### Evaluation Visualizations")
        
        img_col1, img_col2 = st.columns(2)
        
        cm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'images', 'confusion_matrix.png'))
        roc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'images', 'roc_curve.png'))
        fi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'images', 'feature_importance.png'))
        
        with img_col1:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix on Test Dataset")
            else:
                st.info("Confusion matrix plot not found.")
                
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Model Feature Importances")
            else:
                st.info("Feature importance plot not found.")
                
        with img_col2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curve")
            else:
                st.info("ROC Curve plot not found.")
    else:
        st.info("Evaluation report JSON not found. Train the model using the button on single prediction tab or run the training script.")
