# MLOps Assignment with MLflow

## Project Overview
This project demonstrates an MLOps pipeline using MLflow for tracking, DVC for data versioning, and Python scripts for preprocessing, training, and evaluation. The model predicts Boston housing prices using Random Forest.

## Setup Instructions
1. Clone the repo:
   ```bash
   git clone https://github.com/<username>/mlops-mlflow-assignment.git
   cd mlops-mlflow-assignment
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Setup DVC remote if needed:

   ```bash
   dvc remote add -d storage /path/to/remote
   dvc pull
   ```

## Pipeline Walkthrough

1. Preprocess data:

   ```bash
   python src/data_preprocessing.py
   ```
2. Train model:

   ```bash
   python src/model_training.py
   ```
3. Evaluate model (use MLflow run_id):

   ```bash
   python src/model_evaluation.py --run_id <latest_run_id>
   ```
4. Or run full pipeline:

   ```bash
   python src/pipeline.py
   ```

## MLflow Tracking

```bash
mlflow ui
```

Open `http://localhost:5000` to see logged models and metrics.
