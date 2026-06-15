import joblib
import pandas as pd 

# LOAD MODEL 
model_path = r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\02_Models\xgboost_fraud_model.pkl"
model = joblib.load(model_path)

# PREDICTION FUNCTION
def predict_transaction(transaction_data):
    """
    transaction_data:
    Pandas DataFrame containing one transaction
    """

    probability = model.predict_proba(transaction_data)[0][1]
    prediction = model.predict(transaction_data)[0]

    status = ("Fraudulent" if prediction == 1 else "Legitimate")
    return {"fraud_probability": round(probability * 100, 2), "status": status,"prediction": int(prediction),"risk_level": risk_level}

