# Credit Card Fraud Detection System

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning and anomaly detection techniques. The dataset contains over 284,000 transactions with an extreme class imbalance, where fraudulent transactions account for approximately 0.17% of all records.

The objective is to build a robust fraud detection system capable of identifying fraudulent transactions while minimizing false positives and false negatives.

---

## Business Problem

Financial institutions process millions of transactions daily. Even a small percentage of fraudulent transactions can result in significant financial losses and reputational damage.

The challenge lies in detecting rare fraudulent events hidden within a vast number of legitimate transactions.

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

* Dataset exploration and quality assessment
* Missing value analysis
* Class imbalance analysis
* Transaction amount analysis
* Correlation analysis
* Fraud pattern investigation

### 2. Data Preprocessing

* Duplicate removal
* Feature scaling
* Train-test splitting
* Data preparation for modeling

### 3. Imbalanced Data Handling

* Applied SMOTE (Synthetic Minority Oversampling Technique)
* Balanced fraud and non-fraud classes
* Improved model learning capability

### 4. Machine Learning Models

#### Logistic Regression

Used as a baseline model to establish benchmark performance.

#### XGBoost

Used as the primary supervised learning model due to its strong performance on tabular and imbalanced datasets.

#### Isolation Forest

Implemented as an unsupervised anomaly detection model to identify potentially fraudulent transactions without relying on labels.

---

## Model Comparison

Models were evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The comparison framework enabled objective model selection and performance validation.

### Key Observation

XGBoost consistently delivered the best balance between fraud detection capability and false alarm reduction, making it the preferred production model.

---

## Dashboard Features

* Project Overview
* Model Performance Metrics
* Real-Time Fraud Detection
* Model Comparison Visualization
* Risk-Level Classification

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Isolation Forest
* SMOTE
* Matplotlib
* Seaborn
* Streamlit

---

## Project Structure

data/
processed_data/
models/
results/
notebooks/
src/
dashboard/
reports/

---

## Business Impact

This solution demonstrates how machine learning can assist financial institutions in:

* Detecting fraudulent transactions more effectively
* Reducing operational losses
* Improving customer protection
* Supporting fraud investigation teams
* Enhancing risk management processes

---

## Future Enhancements

* Real-time transaction streaming
* Cloud deployment
* Deep learning-based fraud detection
* Graph-based fraud analytics
* Automated alerting system

---

## Author

Aspiring Data Analyst / Data Scientist focused on building end-to-end data and machine learning solutions that solve real-world business problems.
