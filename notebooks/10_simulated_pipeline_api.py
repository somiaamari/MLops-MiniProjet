from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for the frontend

# Simulated database for pipeline statuses
pipeline_status = {
    "feature_engineering": "Not started",
    "training": "Not started",
    "inference": "Not started",
    "monitoring": "Not started",
}

# Simulated predictions and performance metrics
batch_predictions = []
performance_metrics = {
    "accuracy": None,
    "precision": None,
    "recall": None,
}

# Simulated pipeline functions
def run_feature_engineering():
    pipeline_status["feature_engineering"] = "Running"
    print("Running feature engineering pipeline...")
    # Simulate feature processing and save to feature store
    pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}).to_parquet("feature_store.parquet")
    pipeline_status["feature_engineering"] = "Completed"

def run_training():
    pipeline_status["training"] = "Running"
    print("Running training pipeline...")
    # Simulate training a model and saving the model artifact
    pipeline_status["training"] = "Completed"
    performance_metrics["accuracy"] = 0.95
    performance_metrics["precision"] = 0.93
    performance_metrics["recall"] = 0.92

def run_inference(features):
    pipeline_status["inference"] = "Running"
    print("Running inference pipeline...")
    # Simulate inference with dummy predictions
    global batch_predictions
    batch_predictions = [{"input": features[i], "prediction": f"Class {i % 2}"} for i in range(len(features))]
    pipeline_status["inference"] = "Completed"
    return batch_predictions

def run_monitoring():
    pipeline_status["monitoring"] = "Running"
    print("Running monitoring pipeline...")
    # Simulate monitoring by updating metrics or checking drift
    pipeline_status["monitoring"] = "Completed"
    return {"drift_detected": False}

# API endpoints
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": pipeline_status, "metrics": performance_metrics, "predictions": batch_predictions})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get features from the user
        data = request.json
        features = data.get("features", [])
        
        # Run the pipelines
        predictions = run_inference(features)
        monitoring_results = run_monitoring()
        
        return jsonify({
            "predictions": predictions,
            "monitoring": monitoring_results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def trigger_training():
    run_training()
    return jsonify({"message": "Training pipeline triggered!"})

if __name__ == "__main__":
    app.run(debug=True)

