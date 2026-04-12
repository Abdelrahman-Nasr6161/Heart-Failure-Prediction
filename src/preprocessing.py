import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess(filepath, n_components=0.95, apply_pca = True):
    df = pd.read_csv(filepath)
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X = pd.get_dummies(X)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )
    
    # Standardization 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    if apply_pca:
        # PCA Transformation 
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)

        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_val_pca, X_test_pca, y_train.values, y_val.values, y_test.values
    return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv')
    
    print("--- Dataset Split Verification ---")
    print(f"Total Samples: {len(X_train) + len(X_val) + len(X_test)} [Target: 918]")
    print(f"Train set:      {X_train.shape[0]} samples (~70%)")
    print(f"Validation set: {X_val.shape[0]} samples (~10%)")
    print(f"Test set:       {X_test.shape[0]} samples (~20%)")
    print(f"Feature count:  {X_train.shape[1]} (after encoding)")
    print("----------------------------------")