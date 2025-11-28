from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def run_pipeline():
    preprocess_data()
    train_model()
    # Get latest run_id from MLflow UI and replace below
    # run_id = "<latest_run_id>"
    # evaluate_model(run_id=run_id)

if __name__ == "__main__":
    run_pipeline()
