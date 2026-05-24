<div align="center">

# 🏠 Smart Home Activity Recognition — MLOps Project

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> **An end-to-end MLOps pipeline for Smart Home Activity Recognition using Association Rule Mining (Apriori / FP-Growth), Random Forest classification, MLflow experiment tracking, and a Flask REST API — all monitored for model drift.**

</div>

---

## 📑 Table of Contents

- [About The Project](#-about-the-project)
- [Project Structure](#-project-structure)
- [ML Approach](#-ml-approach)
- [MLOps Architecture](#-mlops-architecture)
- [Getting Started](#-getting-started)
- [Running The Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Development History](#-development-history)
- [Team](#-team)

---

## 📖 About The Project

This project applies **MLOps best practices** to the Smart Home domain. A dataset of timestamped smart-home sensor readings (lights, doors, appliances, etc.) is used to:

1. **Discover behavioral patterns** — using Apriori & FP-Growth association rule mining.
2. **Predict user activities** — using a Random Forest classifier with MLflow tracking.
3. **Recommend device states** — a Flask API returns context-aware recommendations.
4. **Monitor model health** — a `RecommendationMonitor` class detects rule drift and triggers retraining.

The entire lifecycle (feature engineering → training → inference → monitoring → retraining) is implemented as a set of REST-callable pipelines.

---

## 📁 Project Structure

```
MLOps/
│
├── app.py                          # ✅ Main Flask application (entry point)
│
├── mlops_smart_home/               # 📦 Structured MLOps package
│   ├── app.py                      #    MLflow-integrated API server (port 5001)
│   ├── config/
│   │   └── config.yaml             #    Central configuration (paths, MLflow, model params)
│   ├── data/
│   │   └── raw/                    #    Place smart_home_dataset.csv here
│   └── src/
│       ├── __init__.py
│       ├── data/
│       │   ├── data_loader.py      #    CSV loading & config parsing
│       │   └── feature_engineering.py  # Feature transformation pipeline
│       ├── models/
│       │   ├── train.py            #    Apriori model training + MLflow logging
│       │   └── predict.py          #    Inference / recommendation logic
│       ├── monitoring/
│       │   └── model_monitoring.py #    Drift detection (RecommendationMonitor)
│       └── utils/
│
├── notebooks/                      # 📓 Development history (iteration scripts)
│   ├── 01_eda_and_association_rules.py
│   ├── 02_apriori_fpgrowth_flask_basics.py
│   ├── 03_flask_visualization.py
│   ├── 04_flask_rules_page.py
│   ├── 05_flask_plot_saving.py
│   ├── 06_flask_optimized.py
│   ├── 07_flask_activity_aware.py
│   ├── 08_mlflow_integration.py
│   ├── 09_ml_pipelines.py
│   └── 10_simulated_pipeline_api.py
│
├── templates/                      # 🌐 Jinja2 HTML templates
│   ├── index.html                  #    Main dashboard
│   ├── index_pipe.html             #    Pipeline monitoring UI
│   └── rules.html                  #    Association rules view
│
├── static/                         # 🎨 CSS stylesheets & generated plots
│   ├── style.css
│   └── styles2.css
│
├── data/                           # 🗄️ Dataset (gitignored — too large for GitHub)
│   └── smart_home_dataset.csv
│
├── docs/                           # 📚 Additional documentation
│
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
└── .gitignore
```

---

## 🤖 ML Approach

### Dataset
- **smart_home_dataset.csv** — Timestamped binary sensor readings for a smart home.
- Columns: `timestamp`, `Activity`, and 20+ binary device-state columns (e.g., `bed`, `tv`, `oven`, `livingLight`, `mainDoorLock` …).

### Association Rule Mining
| Algorithm | Library | Purpose |
|-----------|---------|---------|
| **Apriori** | `mlxtend` | Find frequent device co-activation patterns |
| **FP-Growth** | `mlxtend` | Faster frequent pattern mining on large data |

Key parameters (tunable in `config.yaml`):
- `min_support` — Minimum fraction of transactions containing an itemset
- `min_confidence` — Minimum conditional probability for a rule
- Rules filtered by `lift > 1` to ensure non-trivial associations

### Activity Classification
- **Random Forest Classifier** (`sklearn`) trained on engineered features
- Labels: encoded `Activity` column (sleep, work, leisure, etc.)
- Tracked with **MLflow**: accuracy, model artifact, and parameters logged per run

---

## 🏗️ MLOps Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────────────┐
│  Feature Engineering│  ← feature_engineering.py
│  (normalize, encode)│
└─────────┬───────────┘
          │  Feature Store (.parquet)
          ▼
┌─────────────────────┐
│   Training Pipeline │  ← train.py + MLflow tracking
│   Apriori / RF      │
└─────────┬───────────┘
          │  Model artifacts + Rules
          ▼
┌─────────────────────┐
│  Inference Pipeline │  ← Flask API /recommend endpoint
│  Activity-aware recs│
└─────────┬───────────┘
          │  Predictions
          ▼
┌─────────────────────┐
│ Monitoring Pipeline │  ← RecommendationMonitor (drift detection)
│  KS-test / drift    │     → triggers retraining if needed
└─────────────────────┘
```

All 4 stages are accessible via REST endpoints.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12
- pip

### 1. Clone the repository
```bash
git clone https://github.com/somiaamari/MLops-MiniProjet.git
cd MLops-MiniProjet
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download or place `smart_home_dataset.csv` into the `data/` folder:
```
data/smart_home_dataset.csv
```
> ⚠️ The dataset is **not committed** to Git (90 MB). Place it manually before running.

---

## ▶️ Running The Application

### Main Flask App (root `app.py`)
```bash
python app.py
```
Visit: [http://localhost:5000](http://localhost:5000)

### MLOps Package App (`mlops_smart_home/app.py`)
```bash
cd mlops_smart_home
python app.py
```
Visit: [http://localhost:5001](http://localhost:5001)

### Start MLflow UI (optional)
```bash
mlflow ui
```
Visit: [http://localhost:5000/#/experiments](http://localhost:5000/#/experiments)

---

## 🔌 API Endpoints

### Main App (`app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard with top association rules & chart |
| `POST` | `/recommend` | Get device recommendations based on current state |
| `POST` | `/add_data` | Add a new sensor reading to the dataset |

**Example `/recommend` request:**
```json
POST /recommend
{
  "user_data": {
    "bed": "1",
    "tv": "0",
    "oven": "1",
    "activity": "sleep"
  }
}
```

### MLOps Package App (`mlops_smart_home/app.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/train` | Run full training pipeline + MLflow logging |
| `GET/POST` | `/monitor` | Run drift detection on current vs. reference rules |
| `GET` | `/rules` | Fetch latest generated association rules |

---

## 📓 Development History

The `notebooks/` folder contains the **full iterative development** of this project, from raw EDA to the final production app:

| Script | What it covers |
|--------|---------------|
| `01_eda_and_association_rules.py` | Data loading, preprocessing, EDA, PCA, first Apriori run |
| `02_apriori_fpgrowth_flask_basics.py` | Combining Apriori + FP-Growth, basic Flask `/get_rules` endpoint |
| `03_flask_visualization.py` | Flask with bar chart visualization for rules |
| `04_flask_rules_page.py` | Flask with dedicated `/rules` page (Jinja2 template) |
| `05_flask_plot_saving.py` | Save matplotlib plots as static PNGs for the web UI |
| `06_flask_optimized.py` | Memory optimization (`downcast`), caching plot generation |
| `07_flask_activity_aware.py` | Activity-aware recommendations (rules filtered per activity) |
| `08_mlflow_integration.py` | Full MLflow experiment tracking in the Flask app |
| `09_ml_pipelines.py` | End-to-end pipelines: feature store → training → inference → monitoring |
| `10_simulated_pipeline_api.py` | Simulated pipeline API with status tracking & CORS support |

---

## 👩‍💻 Team

| Name | Role |
|------|------|
| **Amari Soumia** | ML Engineer & MLOps |
| **Berrahou Meriem** | Data Engineering & Flask API |
| **Mehda Nessrin** | Model Monitoring & Evaluation |

---

<div align="center">

**Mini-Project — MLOps Course**  
*Smart Home Activity Recognition & Automation using Association Rule Mining*

</div>
