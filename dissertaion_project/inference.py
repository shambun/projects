from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
import joblib
import re
import pickle
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

import logging

# Configure logging to write to a file (append mode)
logging.basicConfig(
    filename="prediction_debug.log",  # Log file name
    level=logging.DEBUG,  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format with timestamp
    filemode="w",  # Overwrites the log file each time the script runs; use "a" for append mode
)

logging.info("Prediction model started.")  # Initial log message



# Load encoders and vectorizers
with open("data/one_hot_encoder_feb11.pkl", "rb") as f:
    encoder = pickle.load(f)

le = joblib.load("data/complaint_label_encoder_feb11.pkl")
tfidf = joblib.load("data/tfidf_vectorizer_feb11.pkl")

# Load trained model
#model = joblib.load("rf__struct_0205_01.joblib")
model = joblib.load("models/train.joblib")

# Load bin chargeback rate data
bin_chargeback_rate = pd.read_csv("data/bin_chargeback_rates_feb11.csv", index_col=0).squeeze("columns")
bin_country_chargeback_rate = pd.read_csv("data/bin_country_chargeback_rates_feb11.csv", index_col=[0, 1]).squeeze("columns").to_dict()

# Define FastAPI app
app = FastAPI()

# Define input schema
class TransactionInput(BaseModel):
    CustomerID: str
    CardNumber: str
    BankName: str
    CardBrand: str
    CardType: str
    IssuingCountry: str
    TransactionAmount: float
    TransactionDate: str
    Complaint: str
    Feedback: str

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text) if isinstance(text, str) else ""

def preprocess_data(new_data: dict):
    new_data = pd.DataFrame([new_data])
    new_data.fillna("", inplace=True)

    # Extract BIN (first 6 digits of Card Number)
    new_data['BIN'] = new_data['CardNumber'].astype(str).str[:6]

    # Compute BIN risk scores
    new_data['BINCountryRisk_Score'] = new_data.apply(
        lambda row: bin_country_chargeback_rate.get((row['BIN'], row['IssuingCountry']), 0), axis=1
    )

    high_risk_bins = bin_chargeback_rate[bin_chargeback_rate > 0.05].index
    new_data['HighRiskBIN'] = new_data['BIN'].apply(lambda x: 1 if x in high_risk_bins else 0)

    #with open("debug_log.txt", "w") as w:
    #    w.write(f"new data write 1: {new_data.columns.tolist()}\n")
    logging.info(f"new data write 1: {new_data.columns.tolist()}\n")

    # One-hot encode categorical variables
    categorical_columns = ['BankName', 'CardBrand', 'CardType', 'IssuingCountry']
    encoded_new_data = encoder.transform(new_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out())
    #w.write(f"encoded_df write 2: {encoded_df.columns.tolist()}\n")
    logging.info(f"encoded_df write 2: {encoded_df.columns.tolist()}\n")

    # Process complaints and feedback
    new_data['Complaint'] = new_data['Complaint'].apply(clean_text)
    new_data['Feedback'] = new_data['Feedback'].apply(clean_text)

    print("Unique Complaint Categories in Input:", new_data["Complaint"].unique())

    # Encode complaints
    new_data['Complaint_Encoded'] = le.transform(new_data['Complaint'])
    #w.write(f"new_data - Complaint_Encoded- write 3: {new_data.columns.tolist()}\n")
    logging.info(f"new_data write 3: {new_data.columns.tolist()}\n")

    # ✅ Corrected: Use `transform()` instead of `fit_transform()`
    feedback_tfidf = tfidf.transform(new_data['Feedback']).toarray()
    tfidf_df = pd.DataFrame(feedback_tfidf, columns=[f"TFIDF_{i}" for i in range(feedback_tfidf.shape[1])])

    # Reset indices to prevent alignment issues
    new_data.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    #w.write(f"tfidf_df - write 4: {tfidf_df.columns.tolist()}\n")
    logging.info(f"tfidf_df write 4: {tfidf_df.columns.tolist()}\n")

    # Concatenate processed TF-IDF data
    new_data = pd.concat([new_data, tfidf_df], axis=1)

    # Drop original text columns
    new_data.drop(columns=['Feedback', 'Complaint'], inplace=True)

    # Drop unused columns and concatenate processed data
    processed_data = new_data.drop(columns=categorical_columns + ['CustomerID', 'CardNumber', 'TransactionAmount', 'TransactionDate'])
    #final_data = new_data.drop(columns=categorical_columns + ['CustomerID', 'CardNumber', 'TransactionAmount', 'TransactionDate'])
    #w.write(f"processed_data - write 5: {processed_data.columns.tolist()}\n")
    logging.info(f"processed_data write 5: {processed_data.columns.tolist()}\n")
    #if "ChargebackFlag" in final_data.columns:
        #final_data.drop(columns=["ChargebackFlag"], inplace=True)
    # ✅ Fix: Use `tfidf_df` instead of undefined `feedback_df`
    #final_data = pd.concat([processed_data, encoded_df, tfidf_df], axis=1)
    final_data = pd.concat([processed_data, encoded_df], axis=1)

    # Load expected feature names from training
    expected_features = joblib.load("data/expected_feature_names.pkl")

    # Remove "ChargebackFlag" if it exists in the list
    if "ChargebackFlag" in expected_features:
        expected_features.remove("ChargebackFlag")

    # Log for debugging
    logging.info(f"Filtered Expected Features (without ChargebackFlag): {expected_features}\n")

    # Reorder columns to match training order (without ChargebackFlag)
    final_data = final_data.reindex(columns=expected_features, fill_value=0)  # Fill missing features with 0

    # Log final features in prediction data
    logging.info(f"Final Data Columns after Reindexing: {final_data.columns.tolist()}\n")


    # Load the feature names from training (Assuming they were saved)
    #expected_features = joblib.load("expected_feature_names.pkl")  # Ensure this was saved during training
    #logging.info(f"expected_features write 6: {expected_features.columns.tolist()}\n")
    #logging.info(f"Expected features (from training) write 6: {expected_features}\n")

    # Reorder columns to match training order
    #final_data = final_data.reindex(columns=expected_features, fill_value=0)  # Fill missing features with 0
    
    logging.info(f"final_data write 7: {final_data.columns.tolist()}\n")
    print("Final Processed Data Columns:", final_data.columns.tolist())

    #with open("debug_log.txt", "w") as w:
       # w.write(f"Final Data Columns: {final_data.columns.tolist()}\n")



    return final_data


@app.post("/predict")
async def predict_chargeback(transaction: TransactionInput):
    try:
        transaction_dict = transaction.dict()
        final_data = preprocess_data(transaction_dict)
        print("Features in input:", final_data.columns.tolist())
        print("Features expected by model:", model.feature_names_in_)
        #final_data = final_data.reindex(columns=expected_columns, fill_value=0)

        chargeback_prob = model.predict_proba(final_data)[:, 1][0]
        
        threshold = 0.5
        prediction = "High Risk" if chargeback_prob > threshold else "Low Risk"
        
        response = {
            "Customer ID": transaction.CustomerID,
            "Transaction Amount": transaction.TransactionAmount,
            "Chargeback Probability": round(chargeback_prob, 4),
            "Prediction": prediction,
            "Response Timestamp": datetime.now().isoformat()
        }
        
        return response
    except Exception as e:
        return {"error": str(e)}
