from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Create Flask app
app = Flask(__name__)

# Paths
FEATURE_STORE_PATH = "feature_store.parquet"
MODEL_REGISTRY_PATH = "model_registry"
MONITORING_DATA_PATH = "monitoring_data.csv"
RAW_DATA_PATH = "../data/smart_home_dataset.csv"

# Ensure directories exist
os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)

# Load raw data
def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)

# 1. Feature Engineering Pipeline
def feature_engineering_pipeline():
    raw_data = load_raw_data()
    raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], format='%Y-%m-%d %H_%M_%S')
    raw_data.fillna(0, inplace=True)

    # Example feature engineering
    features = raw_data.drop(columns=["Activity", "timestamp"])
    features["activity_encoded"] = pd.factorize(raw_data["Activity"])[0]
    
    # Save features to the feature store
    features.to_parquet(FEATURE_STORE_PATH, index=False)
    print("Feature engineering completed and saved to feature store.")

# 2. Training/Retraining Pipeline
def training_pipeline():
    # Load features from the feature store
    features = pd.read_parquet(FEATURE_STORE_PATH)
    X = features.drop(columns=["activity_encoded"])
    y = features["activity_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model training completed. Accuracy: {accuracy:.2f}")

    # Save model to model registry
    model_version = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(MODEL_REGISTRY_PATH, f"model_{model_version}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to registry: {model_path}")

    # Track with MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("smart_home_ml_experiment")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")
    print("Model and metrics logged to MLflow.")

# 3. Inference Pipeline
def inference_pipeline():
    # Load latest model from the model registry
    model_files = sorted(os.listdir(MODEL_REGISTRY_PATH), reverse=True)
    if not model_files:
        raise Exception("No models found in the registry!")
    latest_model_path = os.path.join(MODEL_REGISTRY_PATH, model_files[0])
    model = joblib.load(latest_model_path)

    # Load features from the feature store
    features = pd.read_parquet(FEATURE_STORE_PATH)
    X = features.drop(columns=["activity_encoded"])
    
    # Predict
    predictions = model.predict(X)
    print("Inference completed. Predictions generated.")

    # Save predictions
    predictions_df = pd.DataFrame({"prediction": predictions})
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv.")

# 4. ML Monitoring Pipeline
def monitoring_pipeline():
    # Example: Check model drift (e.g., feature distributions)
    predictions = pd.read_csv("predictions.csv")
    monitoring_data = pd.DataFrame({"timestamp": datetime.now(), "predictions_mean": predictions.mean()})
    
    # Append monitoring data
    if os.path.exists(MONITORING_DATA_PATH):
        existing_data = pd.read_csv(MONITORING_DATA_PATH)
        monitoring_data = pd.concat([existing_data, monitoring_data], ignore_index=True)
    monitoring_data.to_csv(MONITORING_DATA_PATH, index=False)
    print("Monitoring data updated.")
    
    # Trigger retraining if drift is detected
    if monitoring_data["predictions_mean"].std() > 0.1:  # Example drift condition
        print("Drift detected. Triggering retraining.")
        training_pipeline()

# Routes for triggering pipelines
@app.route('/run_feature_engineering', methods=['POST'])
def run_feature_engineering():
    feature_engineering_pipeline()
    return jsonify({"message": "Feature engineering pipeline executed successfully."})

@app.route('/run_training', methods=['POST'])
def run_training():
    training_pipeline()
    return jsonify({"message": "Training pipeline executed successfully."})

@app.route('/run_inference', methods=['POST'])
def run_inference():
    inference_pipeline()
    return jsonify({"message": "Inference pipeline executed successfully."})

@app.route('/run_monitoring', methods=['POST'])
def run_monitoring():
    monitoring_pipeline()
    return jsonify({"message": "Monitoring pipeline executed successfully."})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

