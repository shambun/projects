import pandas as pd
import re
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import gc
from memory_profiler import profile

# Initialize MLflow
mlflow.set_tracking_uri("http://mlflow-service:5000")
mlflow.set_experiment("Chargeback_Prediction")

def preprocess_data():
    """Preprocess the data: clean, encode, and merge datasets."""
    # Load and clean data
    data_file = "data/historical_consumer_data_0130_02.csv"
    data_df = pd.read_csv(data_file)
    print("Column Count:", len(data_df.columns))
    print(data_df.head())

    def clean_bank_name(name):
        """Removes special characters from bank names."""
        return re.sub(r'[^a-zA-Z0-9 ]', '', name) if isinstance(name, str) else name

    data_df['BankName'] = data_df['BankName'].apply(clean_bank_name)
    data_df['ChargebackFlag'] = data_df['ChargebackFlag'].map({'No': 0, 'Yes': 1})
    data_df['BIN'] = data_df['CardNumber'].astype(str).str[:6]

    # Compute BIN-based features
    bin_chargeback_rate = data_df.groupby('BIN')['ChargebackFlag'].mean()
    bin_chargeback_rate.to_csv("data/bin_chargeback_rates_feb11.csv")

    bin_country_chargeback_rate = data_df.groupby(['BIN', 'IssuingCountry'])['ChargebackFlag'].mean()
    bin_country_chargeback_rate.to_csv("data/bin_country_chargeback_rates_feb11.csv")
    data_df['BINCountryRisk_Score'] = data_df.apply(lambda row: bin_country_chargeback_rate.get((row['BIN'], row['IssuingCountry']), 0), axis=1)

    high_risk_bins = bin_chargeback_rate[bin_chargeback_rate > 0.05].index
    data_df['HighRiskBIN'] = data_df['BIN'].apply(lambda x: 1 if x in high_risk_bins else 0)

    # One-Hot Encoding
    categorical_columns = ['BankName', 'CardBrand', 'CardType', 'IssuingCountry']
    data_df[categorical_columns] = data_df[categorical_columns].map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit(data_df[categorical_columns])

    with open("data/one_hot_encoder_feb11.pkl", "wb") as f:
        pickle.dump(encoder, f)

    encoded_data = encoder.transform(data_df[categorical_columns])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

    processed_data = data_df.drop(columns=categorical_columns + ['CustomerID', 'CardNumber', 'TransactionAmount', 'ChargebackReasonCode', 'TransactionDate'])
    final_data = pd.concat([processed_data, encoded_df], axis=1)

    # Load unstructured data
    file_path = "data/synthetic_sentiment_data_0223_10k.csv"
    df = pd.read_csv(file_path)
    df = df[['BIN', 'Complaint', 'Feedback', 'ChargebackFlag']]
    df.dropna(inplace=True)

    le = LabelEncoder()
    df['Complaint_Encoded'] = le.fit_transform(df['Complaint'])

    tfidf = TfidfVectorizer(max_features=100)
    feedback_tfidf = tfidf.fit_transform(df['Feedback']).toarray()
    tfidf_df = pd.DataFrame(feedback_tfidf, columns=[f"TFIDF_{i}" for i in range(feedback_tfidf.shape[1])])

    joblib.dump(le, "data/complaint_label_encoder_feb11.pkl")
    joblib.dump(tfidf, "data/tfidf_vectorizer_feb11.pkl")
    df.to_csv("data/preprocessed_complaints.csv", index=False)

    df.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, tfidf_df], axis=1)
    df.drop(columns=['Feedback', 'Complaint'], inplace=True)

    df["BIN"] = df["BIN"].astype(str)
    final_data["BIN"] = final_data["BIN"].astype(str)

    merged_df = pd.merge(final_data, df, on='BIN', how='inner', suffixes=('_structured', '_unstructured'))
    merged_df['ChargebackFlag'] = merged_df[['ChargebackFlag_structured', 'ChargebackFlag_unstructured']].max(axis=1)
    merged_df.drop(columns=['ChargebackFlag_structured', 'ChargebackFlag_unstructured'], inplace=True)

    joblib.dump(list(merged_df.columns), "data/expected_feature_names.pkl")

    # Train-test split and SMOTE
    X = merged_df.drop(columns=['ChargebackFlag'])
    y = merged_df['ChargebackFlag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test

def train_model(X_train, y_train):
    """Train the RandomForestClassifier model."""
    rf_model = RandomForestClassifier(
        n_estimators=2000,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_split=10
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics to MLflow."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.47
    y_pred_new = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_new)
    roc_auc = roc_auc_score(y_test, y_pred_new)
    classification_rep = classification_report(y_test, y_pred_new)
    conf_matrix = confusion_matrix(y_test, y_pred_new)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log classification report and confusion matrix as artifacts
    with open("classification_report.txt", "w") as f:
        f.write(classification_rep)
    mlflow.log_artifact("classification_report.txt")

    with open("confusion_matrix.txt", "w") as f:
        f.write(str(conf_matrix))
    mlflow.log_artifact("confusion_matrix.txt")

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Save the model locally
    joblib.dump(model, "models/train.joblib")
    print("\nModel saved as 'train.joblib'.")

    # Plot feature importance
    feature_importances = model.feature_importances_
    feature_names = X_test.columns
    indices = np.argsort(feature_importances)[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.title("Top 10 Feature Importance (Random Forest)")
    plt.savefig("feature_importance.png")
    plt.close()

    # Log the feature importance plot as an artifact
    mlflow.log_artifact("feature_importance.png")

    print("✅ Training and evaluation completed! Results logged to MLflow.")

@profile
def train():
    """Main function to handle the training workflow."""
    try:
        # Start an MLflow run
        run = mlflow.start_run()

        # Log parameters
        mlflow.log_param("n_estimators", 2000)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data()

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        # Clear memory
        del model
        gc.collect()
    finally:
        # Ensure the run is terminated
        mlflow.end_run()

# Run the training function
if __name__ == "__main__":
    train()