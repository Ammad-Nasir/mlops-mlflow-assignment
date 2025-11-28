import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path='data/raw_data.csv', output_path='data/processed_data.csv'):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['PRICE'] = y

    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
