from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def run_pipeline():
    print("Step 1: Preprocessing data...")
    preprocess_data()

    print("Step 2: Training model...")
    train_model()

    # After training, get latest MLflow run ID from UI or code
    import mlflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")  # Default experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time desc"], max_results=1)
    latest_run_id = runs[0].info.run_id
    print(f"Latest MLflow run ID: {latest_run_id}")

    print("Step 3: Evaluating model...")
    evaluate_model(latest_run_id)

if __name__ == "__main__":
    run_pipeline()
