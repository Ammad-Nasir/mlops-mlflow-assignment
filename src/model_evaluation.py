import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_model(run_id, data_path="data/processed_data.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run data_preprocessing.py first.")
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model from MLflow using run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation Metrics:\nMSE: {mse}\nR2: {r2}")

if __name__ == "__main__":
    # Replace <run_id> with the actual run ID from MLflow UI
    run_id = "<run_id_here>"
    evaluate_model(run_id)
