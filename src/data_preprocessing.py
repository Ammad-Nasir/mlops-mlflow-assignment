import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path="data/raw_data.csv", output_path="data/processed_data.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Make sure your dataset exists.")
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine scaled features with target
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['PRICE'] = y

    # Save preprocessed dataset
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()
