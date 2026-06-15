import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

X_train = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_train.csv")

X_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_test.csv")

y_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_test.csv").squeeze()

# CREATE ISOLATION FOREST
iso_model = IsolationForest(contamination=0.0017,random_state=42,n_estimators=200)
iso_model.fit(X_train)

predictions = iso_model.predict(X_test)
predictions = [1 if x == -1 else 0 for x in predictions]

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print(cm)

auc = roc_auc_score(y_test, predictions)
print("ROC-AUC:", auc)

joblib.dump(iso_model, r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models\isolation_forest.pkl")

xgb_prob= 0.85
iso_flag = 1
if xgb_prob > 0.70 or iso_flag == 1:
    result = "Fraud"
else:
    result = "Legitimate"

print(result)

results = {"Model": "Isolation Forest", "ROC_AUC": auc}
print(results)