import pandas as pd
import mlflow

def engineer_features(df):
    """Feature engineering pipeline with MLflow tracking"""
    with mlflow.start_run(run_name="feature_engineering"):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H_%M_%S')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Log feature engineering params
        mlflow.log_param("features_created", ["hour", "day_of_week"])
        
        return df
