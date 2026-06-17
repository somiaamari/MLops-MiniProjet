import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from scipy.stats import ks_2samp
from sklearn.metrics import silhouette_score

# class RecommendationMonitor:
#     def __init__(self, config):
#         self.config = config
#         self.support_threshold = config.get('monitoring', {}).get('support_threshold', 0.1)
#         self.confidence_threshold = config.get('monitoring', {}).get('confidence_threshold', 0.5)
#         self.history = []

#     def calculate_rule_metrics(self, rules_df):
#         """Calculate metrics for association rules"""
#         metrics = {
#             'avg_support': rules_df['support'].mean(),
#             'avg_confidence': rules_df['confidence'].mean(),
#             'rules_count': len(rules_df),
#             'high_confidence_rules': len(rules_df[rules_df['confidence'] > self.confidence_threshold]),
#             'coverage': len(set(rules_df['antecedent'].sum() + rules_df['consequent'].sum()))
#         }
#         return metrics

#     def detect_pattern_drift(self, current_rules, reference_rules):
#         """Detect if there's significant drift in rule patterns"""
#         current_metrics = self.calculate_rule_metrics(current_rules)
#         reference_metrics = self.calculate_rule_metrics(reference_rules)
        
#         # Calculate drift scores
#         drift_scores = {
#             'support_drift': abs(current_metrics['avg_support'] - reference_metrics['avg_support']),
#             'confidence_drift': abs(current_metrics['avg_confidence'] - reference_metrics['avg_confidence']),
#             'rules_count_drift': abs(current_metrics['rules_count'] - reference_metrics['rules_count']) / max(reference_metrics['rules_count'], 1),
#             'coverage_drift': abs(current_metrics['coverage'] - reference_metrics['coverage']) / max(reference_metrics['coverage'], 1)
#         }
        
#         return drift_scores

#     def monitor_rules(self, current_rules, reference_rules):
#         """Main monitoring function"""
#         with mlflow.start_run(run_name="rules_monitoring"):
#             # Calculate drift metrics
#             drift_scores = self.detect_pattern_drift(current_rules, reference_rules)
            
#             # Log metrics to MLflow
#             mlflow.log_metrics(drift_scores)
#             mlflow.log_metric("timestamp", datetime.now().timestamp())
            
#             # Store monitoring history
#             self.history.append({
#                 'timestamp': datetime.now(),
#                 'metrics': drift_scores
#             })
            
#             # Determine if retraining is needed
#             needs_retraining = (
#                 drift_scores['support_drift'] > self.support_threshold or
#                 drift_scores['confidence_drift'] > self.confidence_threshold
#             )
            
#             return {
#                 'drift_scores': drift_scores,
#                 'needs_retraining': needs_retraining,
#                 'monitoring_timestamp': datetime.now().isoformat()
#             }

#     def get_monitoring_history(self):
#         """Return monitoring history for visualization"""
#         return pd.DataFrame(self.history)

class RecommendationMonitor:
    def __init__(self, config):
        self.config = config
        self.support_threshold = config.get('monitoring', {}).get('support_threshold', 0.1)
        self.confidence_threshold = config.get('monitoring', {}).get('confidence_threshold', 0.5)
        self.history = []
        self.rules_history = []

    def calculate_rule_metrics(self, rules_df):
        """Calculate metrics for association rules."""
        metrics = {
            'avg_support': rules_df['support'].mean(),
            'avg_confidence': rules_df['confidence'].mean(),
            'rules_count': len(rules_df),
            'high_confidence_rules': len(rules_df[rules_df['confidence'] > self.confidence_threshold]),
            'coverage': len(set(rules_df['antecedent'].sum() + rules_df['consequent'].sum()))
        }
        return metrics

    def detect_pattern_drift(self, current_rules, reference_rules):
        """Detect if there's significant drift in rule patterns."""
        current_metrics = self.calculate_rule_metrics(current_rules)
        reference_metrics = self.calculate_rule_metrics(reference_rules)

        # Calculate drift scores
        drift_scores = {
            'support_drift': abs(current_metrics['avg_support'] - reference_metrics['avg_support']),
            'confidence_drift': abs(current_metrics['avg_confidence'] - reference_metrics['avg_confidence']),
            'rules_count_drift': abs(current_metrics['rules_count'] - reference_metrics['rules_count']) / max(reference_metrics['rules_count'], 1),
            'coverage_drift': abs(current_metrics['coverage'] - reference_metrics['coverage']) / max(reference_metrics['coverage'], 1),
        }

        return drift_scores

    def monitor_rules(self, current_rules, reference_rules):
        """Main monitoring function."""
        with mlflow.start_run(run_name="rules_monitoring"):
            # Calculate drift metrics
            drift_scores = self.detect_pattern_drift(current_rules, reference_rules)

            # Log metrics to MLflow
            mlflow.log_metrics(drift_scores)
            mlflow.log_metric("timestamp", datetime.now().timestamp())

            # Store monitoring history
            self.history.append({
                'timestamp': datetime.now(),
                'metrics': drift_scores
            })

            # Determine if retraining is needed
            needs_retraining = (
                drift_scores['support_drift'] > self.support_threshold or
                drift_scores['confidence_drift'] > self.confidence_threshold
            )

            return {
                'drift_scores': drift_scores,
                'needs_retraining': needs_retraining,
                'monitoring_timestamp': datetime.now().isoformat()
            }
    def get_monitoring_history(self):
        """Return monitoring history for visualization"""
        return pd.DataFrame(self.history)


    def get_rules(self):
        """Return the latest set of rules generated"""
        if self.rules_history:
            return self.rules_history[-1]  # Return the most recent rules
        else:
            return {"message": "No rules have been generated yet"}
