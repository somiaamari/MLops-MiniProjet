from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

# import mlflow
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def train_apriori_model(data, config):
    """
    Train an Apriori model to generate frequent itemsets and association rules.
    
    Args:
        data (pd.DataFrame): Input data containing binary-encoded transaction data.
        config (dict): Configuration dictionary containing Apriori parameters.
        
    Returns:
        dict: A dictionary containing frequent itemsets, association rules, and metrics.
    """
    with mlflow.start_run(run_name="apriori_training"):
        try:
            # Preprocess data for Apriori
            data_for_apriori = data.drop(columns=["Activity", "timestamp"], errors='ignore')
            transactions = data_for_apriori.apply(
                lambda row: [col for col, value in row.items() if value == 1], axis=1
            ).tolist()
            
            # Encode transactions for Apriori
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

            # Generate frequent itemsets
            frequent_itemsets = apriori(
                df_encoded,
                min_support=config['apriori']['min_support'],
                use_colnames=True
            )
            num_itemsets = len(frequent_itemsets)

            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric=config['apriori']['metric'],
                min_threshold=config['apriori']['min_threshold'],
                num_itemsets=num_itemsets
            )

            # Metrics for logging
            metrics = {
                "num_itemsets": num_itemsets,
                "num_rules": len(rules),
                "avg_support": frequent_itemsets['support'].mean() if not frequent_itemsets.empty else 0,
                "avg_confidence": rules['confidence'].mean() if not rules.empty else 0,
            }

            # Log metrics and results
            mlflow.log_metrics(metrics)
            mlflow.log_param("min_support", config['apriori']['min_support'])
            mlflow.log_param("metric", config['apriori']['metric'])
            mlflow.log_param("min_threshold", config['apriori']['min_threshold'])

            # Save frequent itemsets and rules as artifacts
            frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
            rules.to_csv("association_rules.csv", index=False)
            mlflow.log_artifact("frequent_itemsets.csv")
            mlflow.log_artifact("association_rules.csv")

            return {
                "frequent_itemsets": frequent_itemsets,
                "rules": rules,
                "metrics": metrics
            }

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise e
