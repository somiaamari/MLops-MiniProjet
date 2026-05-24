# 📚 Project Overview & Development Notes

## Development Phases

This project was built iteratively. The `notebooks/` folder captures every step:

### Phase 1 — Exploration (`01_eda_and_association_rules.py`)
- Loaded and explored the smart home dataset
- Applied `LabelEncoder` + `MinMaxScaler` preprocessing
- Ran first Apriori experiment with `mlxtend`
- Visualized: activity frequency, correlation heatmap, PCA scatter

### Phase 2 — Flask Integration (`02–04`)
- Wrapped Apriori logic inside a Flask app
- Added FP-Growth as an alternative algorithm
- Built Jinja2 templates for rule visualization
- Created `/get_rules` and `/rules` endpoints

### Phase 3 — Optimization (`05–07`)
- Added memory optimization via pandas `downcast`
- Cached plot generation (skip if file exists)
- Made recommendations **activity-aware** (filter rules per activity label)
- Proper error handling with `try/except` + JSON error responses

### Phase 4 — MLOps Integration (`08–10`)
- Added MLflow experiment tracking in the Flask app
- Built 4-stage pipeline: Feature Engineering → Training → Inference → Monitoring
- Created `pipelines2.py` — a REST API for pipeline status & triggering

### Phase 5 — Structured Package (`mlops_smart_home/`)
- Refactored into a clean Python package structure
- Centralized config via `config/config.yaml`
- `RecommendationMonitor` class for drift detection (KS-test)
- Full MLflow logging at every stage
- Runs on port **5001** (separate from root app on port 5000)

---

## Key Design Decisions

### Why Apriori over a supervised classifier?
The dataset contains **binary device states** — this naturally maps to a market-basket problem. Association rules let us answer: *"Given these devices are ON, what else should be ON?"* without labeled training data per recommendation.

### Why MLflow?
MLflow tracks every experiment run — support thresholds, confidence levels, number of rules generated — making it easy to compare runs and reproduce results.

### Why two Flask apps?
- `app.py` (root) — Simple, fast, single-file app for demo/deployment
- `mlops_smart_home/app.py` — Production-grade, modular, MLflow-integrated

---

## Notes on `Smart-Home-project/`
This is a **nested Git repository** (has its own `.git`). It is an older snapshot of the project. It is kept for reference but should **not** be pushed as part of the main repo. Consider adding it as a git submodule or removing it before a clean push.

To ignore it completely:
```
# Add to .gitignore
Smart-Home-project/
```
