# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Define the features the model can observe 
### This varaibles will become the HMM emissions

observed_columns = ["transaction_volume","cross_border_ratio","structuring_score"]

def split_accounts(df, random_state=42):

    account_ids = df["account_id"].unique()

    train_accounts, temp_accounts = train_test_split(account_ids, train_size=0.70, random_state=random_state)
    val_accounts, test_accounts = train_test_split(temp_accounts, train_size=0.70, random_state=random_state)

    train_df = df[df["account_id"].isin(train_accounts)].copy()
    val_df = df[df["account_id"].isin(val_accounts)].copy()
    test_df = df[df["account_id"].isin(test_accounts)].copy()

    return train_df, val_df, test_df

def fit_transform_train(df):
    result = df.copy()
    features = result[observed_columns].copy()
    ###log1p is used because
    ###1. If there is 0 normal log can give inf values which then gives errors in training
    ###2. It helps reduce outliers.
    ### For future work, Standard scaler could be replaced with Robustscaler. 
    features["transaction_volume"] = np.log1p(features["transaction_volume"])

    ### Put everything on the same scale
    scaler = StandardScaler()
    result[["transaction_volume_log","cross_border_ratio_scaled","structuring_score_scaled"]] = scaler.fit_transform(features)
    return result, scaler

def fit_transform_val_test(df, scaler):

    transform_df_val_test = df.copy()
    
    features = transform_df_val_test[observed_columns].copy()
    ###log1p is used because
    ###1. If there is 0 normal log can give inf values which then gives errors in training
    ###2. It helps reduce outliers.
    ### For future work, Standard scaler could be replaced with Robustscaler. 
    features["transaction_volume"] = np.log1p(features["transaction_volume"])
    ### Put everything on the same scale
    scaler = StandardScaler()
    transform_df_val_test[["transaction_volume_log","cross_border_ratio_scaled","structuring_score_scaled"]] = scaler.fit_transform(features)
    return transform_df_val_test

def build_sequences(df):
    """
    Build one observation sequence per account.

    """
    ### save each account time periods
    sequences = []
    ### use the same observed columns 
    sequence_columns = ["transaction_volume_log", "cross_border_ratio_scaled","structuring_score_scaled"]
    ###Order each account and order their time period so time periods are not mixed and the HMM can properly identify them
    df_sorted = df.sort_values(["account_id", "time"])

    for account_id, account_df in df_sorted.groupby("account_id", sort=True):
        ### obtain each account observed columns and transform it to a numpy matrix
        sequence = account_df[sequence_columns].to_numpy(dtype=np.float64)
        sequences.append(sequence)

    return sequences

def preprocess_data(df, random_state=42):
    """
    Complete preprocessing pipeline.

    Returns:
        train_df, val_df, test_df
        train_sequences, val_sequences, test_sequences
        scaler
    """

    train_df, val_df, test_df = split_accounts(df, random_state=random_state)

    train_df, scaler = fit_transform_train(train_df)

    val_df = fit_transform_val_test(val_df, scaler)
    test_df = fit_transform_val_test(test_df, scaler)

    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    test_sequences = build_sequences(test_df)

    return train_df, val_df,test_df, train_sequences, val_sequences, test_sequences, scaler
    