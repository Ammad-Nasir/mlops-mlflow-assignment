import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def train_model(data_path="data/processed_data.csv", n_estimators=100, max_depth=5):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run data_preprocessing.py first.")

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run() as run:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f"Model trained and logged to MLflow. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
