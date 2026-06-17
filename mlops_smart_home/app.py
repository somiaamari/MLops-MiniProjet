from flask import Flask, request, jsonify
import mlflow
from src.data.data_loader import load_raw_data, load_config
from src.data.feature_engineering import engineer_features
from src.models.train import train_apriori_model
from src.monitoring.model_monitoring import RecommendationMonitor
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from flask_cors import CORS

app = Flask(__name__)
config = load_config()
CORS(app)  

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

@app.route('/train', methods=["GET", "POST"])
def train_pipeline():
    try:
        # Load and process data
        raw_data = load_raw_data()
        processed_data = engineer_features(raw_data)
        
        # Prepare data for Apriori
        data_for_apriori = processed_data.drop(columns=["Activity", "timestamp"], errors="ignore")
        transactions = data_for_apriori.apply(
            lambda row: [col for col, value in row.items() if value == 1], axis=1
        ).tolist()

        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Generate frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
        frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: list(x))  # Convert itemsets from frozenset to list

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets= len(frequent_itemsets))
        rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))  # Convert antecedents to list
        rules["consequents"] = rules["consequents"].apply(lambda x: list(x))  # Convert consequents to list

        # Metrics for evaluation
        metrics = {
            "num_itemsets": len(frequent_itemsets),
            "num_rules": len(rules),
            "avg_support": frequent_itemsets["support"].mean() if not frequent_itemsets.empty else 0,
            "avg_confidence": rules["confidence"].mean() if not rules.empty else 0,
        }

        # Return JSON response
        return jsonify({
            "status": "success",
            "metrics": metrics,
            "frequent_itemsets": frequent_itemsets.to_dict(orient="records"),
            "rules": rules.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/monitor', methods=['GET','POST'])
# def monitor_pipeline():
#     try:
#         monitor = RecommendationMonitor(config)
#         drift_detected = monitor.check_drift(
#             pd.read_csv(config['data']['predictions_path']),
#             pd.read_csv(config['data']['processed_data_path'])
#         )
        
#         return jsonify({
#             "status": "success",
#             "drift_detected": drift_detected
#         })
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/monitor', methods=['GET', 'POST'])
def monitor_pipeline():
    try:
        # Initialize the RecommendationMonitor
        monitor = RecommendationMonitor(config)

        # Simulate reference rules (instead of reading from a file)
        reference_rules = simulate_reference_rules()  # Function to generate dummy or static data

        # Simulate current rules (instead of relying on predictions)
        current_rules = simulate_current_rules()  # Function to generate dummy or real-time rules

        # Monitor and detect drift
        monitoring_results = monitor.monitor_rules(current_rules, reference_rules)

        # Ensure all data is JSON serializable
        monitoring_results = {
            key: (
                float(value) if isinstance(value, np.float64) else
                bool(value) if isinstance(value, np.bool_) else
                value
            )
            for key, value in monitoring_results.items()
        }

        return jsonify({
            "status": "success",
            "drift_scores": monitoring_results['drift_scores'],
            "needs_retraining": monitoring_results['needs_retraining'],
            "monitoring_timestamp": monitoring_results['monitoring_timestamp']
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Helper functions
def simulate_reference_rules():
    """Generate reference rules with dummy data"""
    data = {
        'antecedent': [['A'], ['B'], ['C']],
        'consequent': [['D'], ['E'], ['F']],
        'support': [0.3, 0.2, 0.1],
        'confidence': [0.8, 0.7, 0.6]
    }
    return pd.DataFrame(data)


def simulate_current_rules():
    """Generate current rules with dummy or dynamic data"""
    data = {
        'antecedent': [['A'], ['B'], ['G']],
        'consequent': [['D'], ['E'], ['H']],
        'support': [0.25, 0.15, 0.05],
        'confidence': [0.75, 0.65, 0.55]
    }
    return pd.DataFrame(data)

@app.route('/rules', methods=['GET'])
def get_rules():
    try:
        # Initialize the RecommendationMonitor
        monitor = RecommendationMonitor(config)

        # Fetch the latest rules
        rules_data = monitor.get_rules()

        # Convert rules data to JSON serializable format
        rules_data = {
            key: (
                float(value) if isinstance(value, np.float64) else
                bool(value) if isinstance(value, np.bool_) else
                value
            )
            for key, value in rules_data.items()
        }

        return jsonify({
            "status": "success",
            "rules": rules_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
