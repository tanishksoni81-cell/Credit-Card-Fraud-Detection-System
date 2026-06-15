import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score)

X_train = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_train.csv")
X_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_test.csv")
y_train = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_train.csv").squeeze()
y_test = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_test.csv").squeeze()
print(y_train.value_counts())

smote = SMOTE(sampling_strategy = 'auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.value_counts())

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_smote, y_train_smote)
y_pred = baseline_model.predict(X_test)
y_prob = baseline_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", auc)

#joblib.dump(baseline_model,r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models/baseline_logistic.pkl")

results = {
    "roc_auc": auc
}

print(results)

# XGBoost + Hyperparameter Tuning

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_smote, y_train_smote)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))

auc_xgb = roc_auc_score(y_test, y_pred_xgb)
print("ROC-AUC:", auc_xgb)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    }

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='recall',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_smote, y_train_smote)
print(search.best_params_)
best_xgb = search.best_estimator_

y_pred_best = best_xgb.predict(X_test)
y_prob_best = best_xgb.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_best))
auc_best = roc_auc_score(y_test, y_prob_best)
print(auc_best)

feature_importance = pd.DataFrame({'Feature': X_train.columns,'Importance': best_xgb.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

top10 = feature_importance.head(10)
plt.figure(figsize=(10,6))
plt.barh(top10['Feature'],top10['Importance'])
plt.title("Top 10 Important Features")
plt.show()

import joblib

joblib.dump(best_xgb,r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models/xgboost_fraud_model.pkl"
)