
```markdown
# MLOps MLflow Assignment

## Project Overview
This project implements a complete MLOps pipeline for predicting Boston housing prices using a Random Forest Regressor.  
It demonstrates **data versioning with DVC**, **model training and tracking with MLflow**, and **CI/CD automation using Jenkins**.

---

## Project Structure
```

mlops-mlflow-assignment/
│
├── data/                       # Raw and processed data
├── dvc_storage/                # DVC remote storage folder
├── src/                        # Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── pipeline.py
├── requirements.txt            # Python dependencies
├── Jenkinsfile                 # Jenkins CI/CD pipeline
├── README.md                   # This file
└── .gitignore                  # Git ignore rules

````

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Ammad-Nasir/mlops-mlflow-assignment.git
cd "mlops 4"
````

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Initialize DVC and set up remote storage**

```bash
dvc init
dvc remote add -d storage ./dvc_storage
dvc add data/raw_data.csv
dvc push
```

4. **Run the ML pipeline**

```bash
python src/pipeline.py
```

5. **View MLflow UI**

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser to see runs, metrics, and logged models.

---

## Pipeline Walkthrough

1. **Preprocessing** (`data_preprocessing.py`)

   * Scales features and splits data into training/testing sets.
   * Saves processed dataset to `data/processed_data.csv`.

2. **Model Training** (`model_training.py`)

   * Trains a Random Forest Regressor.
   * Logs model, parameters, and metrics to MLflow.

3. **Model Evaluation** (`model_evaluation.py`)

   * Evaluates the latest MLflow run on the test set.
   * Prints MSE and R² metrics.

4. **Full Pipeline** (`pipeline.py`)

   * Runs preprocessing → training → evaluation in sequence.
   * Automatically uses the latest MLflow run ID.

5. **CI/CD** (`Jenkinsfile`)

   * Automates checkout, dependency installation, and pipeline execution.

---

## Notes

* MLflow currently logs runs to the local filesystem (`mlruns/`).
* Dataset is versioned with DVC.
* Metrics (MSE, R²) and model artifacts are available in the MLflow UI.

```

---

This README covers:

- **Project overview**  
- **Directory structure**  
- **Setup instructions**  
- **Pipeline explanation**  
- **CI/CD notes**  

It’s ready to push to your GitHub repo as Task 5 deliverable.  


```
