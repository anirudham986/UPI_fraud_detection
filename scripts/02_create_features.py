#!/usr/bin/env python3
"""
Script 2: Create features from raw data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.feature_engineer import FeatureEngineer
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

def main():
    print("="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)

    print("\nLoading data...")
    df = pd.read_csv('data/raw/synthetic_upi_transactions.csv')
    print(f"Loaded {len(df):,} transactions")

    print("\nCreating features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)

    feature_columns = engineer.get_feature_columns()
    print(f"Created {len(feature_columns)} features")

    X = df_features[feature_columns].copy()
    y = df_features['is_fraud'].copy()

    X = X.fillna(X.mean())

    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\nSaving processed data...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("✅ Feature engineering complete!")
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate: {y_test.mean():.2%}")

    feature_list_path = 'data/processed/feature_columns.txt'
    with open(feature_list_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    print(f"\nFeature list saved to: {feature_list_path}")

if __name__ == "__main__":
    main()