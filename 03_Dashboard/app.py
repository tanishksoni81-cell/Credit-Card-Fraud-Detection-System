import streamlit as st
import pandas as pd
import plotly.express as px
import importlib.util
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# =====================================
# PROJECT ROOT
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================
# LOAD PREDICT MODULE
# =====================================

predict_file = os.path.join(
    BASE_DIR,
    "05_src",
    "05_predict.py"
)

try:
    spec = importlib.util.spec_from_file_location(
        "predict_module",
        predict_file
    )

    predict_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(predict_module)

    predict_transaction = predict_module.predict_transaction

except Exception as e:

    st.error(f"Error loading predict.py:\n{e}")
    st.stop()

# =====================================
# LOAD RESULTS
# =====================================

results_file = os.path.join(
    BASE_DIR,
    "03_results",
    "model_comparison.csv"
)

try:

    results = pd.read_csv(results_file)

except Exception as e:

    st.error(f"Error loading model_comparison.csv:\n{e}")
    st.stop()

# =====================================
# TITLE
# =====================================

st.title("💳 Credit Card Fraud Detection Dashboard")

# =====================================
# SIDEBAR
# =====================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Fraud Detection",
        "Model Comparison"
    ]
)

# =====================================
# OVERVIEW
# =====================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        "284,807"
    )

    col2.metric(
        "Fraud Cases",
        "492"
    )

    col3.metric(
        "Fraud Rate",
        "0.17%"
    )

    col4.metric(
        "Best Model",
        "XGBoost"
    )

    st.success(
        "Fraud detection system built using XGBoost, SMOTE, and Isolation Forest."
    )

# =====================================
# MODEL PERFORMANCE
# =====================================

elif page == "Model Performance":

    st.header("Model Performance Metrics")

    st.dataframe(
        results,
        use_container_width=True
    )

# =====================================
# FRAUD DETECTION
# =====================================

elif page == "Fraud Detection":

    st.header("Real-Time Fraud Detection")

    st.info(
        "Upload a CSV file containing transaction records."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            if st.button("Run Detection"):

                result = predict_transaction(
                    df.iloc[[0]]
                )

                st.subheader("Prediction Result")

                st.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability']}%"
                )

                st.metric(
                    "Risk Level",
                    result["risk_level"]
                )

                if result["status"] == "Fraudulent":

                    st.error(
                        f"Status: {result['status']}"
                    )

                else:

                    st.success(
                        f"Status: {result['status']}"
                    )

        except Exception as e:

            st.error(
                f"Prediction Error:\n{e}"
            )

# =====================================
# MODEL COMPARISON
# =====================================

elif page == "Model Comparison":

    st.header("Model Comparison")

    metrics = [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]

    selected_metric = st.selectbox(
        "Select Metric",
        metrics
    )

    fig = px.bar(
        results,
        x="Model",
        y=selected_metric,
        title=f"{selected_metric} Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        results,
        use_container_width=True
    )