import pandas as pd
import yaml
from pathlib import Path

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def load_raw_data():
    config = load_config()
    return pd.read_csv(config['data']['raw_data_path'])