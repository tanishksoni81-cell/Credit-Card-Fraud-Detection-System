import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# LOAD TEST DATA
X_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_test.csv")
y_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_test.csv").squeeze()

# LOAD MODELS
baseline_model = joblib.load(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models/baseline_logistic.pkl")
xgb_model = joblib.load(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models/xgboost_fraud_model.pkl")
iso_model = joblib.load(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models/isolation_forest.pkl")

# LOGISTIC REGRESSION
y_pred_lr = baseline_model.predict(X_test)
y_prob_lr = baseline_model.predict_proba(X_test)[:,1]

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

iso_raw = iso_model.predict(X_test)
y_pred_iso = np.where(iso_raw == -1, 1, 0)

results = pd.DataFrame({"Model":["Logistic Regression","XGBoost","Isolation Forest"],
    "Precision":[precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_xgb),
        precision_score(y_test, y_pred_iso)],

    "Recall":[recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_iso)],
    "F1 Score":[f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_xgb),
        f1_score(y_test, y_pred_iso)],
    "ROC AUC":[roc_auc_score(y_test,y_prob_lr),
        roc_auc_score(y_test, y_prob_xgb),
        roc_auc_score(y_test, y_pred_iso)]})

# DISPLAY RESULTS
print("\nMODEL COMPARISON\n")
print(results.round(4))

# SAVE RESULTS
results.to_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\03_results/model_comparison.csv",index=False)

# BAR CHART
metrics = [
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

for metric in metrics:

    plt.figure(figsize=(8,5))
    sns.barplot(data=results, x="Model", y=metric)
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\03_results\{metric}.png")
    plt.show()

# CONFUSION MATRICES
models = {"Logistic Regression": y_pred_lr,"XGBoost":y_pred_xgb,"Isolation Forest":y_pred_iso}

for name, preds in models.items():

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_CM.png")
    plt.show()

# BEST MODEL
best_model = results.sort_values(by="F1 Score",ascending=False)
print("\nBest Model Based on F1 Score:\n")
print(best_model.iloc[0])