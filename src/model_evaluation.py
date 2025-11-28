import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(data_path='data/processed_data.csv', run_id=None):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if run_id is None:
        print("Error: Please provide MLflow run_id to load the model")
        return

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest_model")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation results - MSE: {mse}, R2: {r2}")
