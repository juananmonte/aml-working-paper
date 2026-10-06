# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Define the features the model can observe 
EMISSION_COLUMNS = ["transaction_volume","cross_border_ratio","structuring_score"]

def split_accounts(df, train_size=0.70, val_size=0.20, test_size=0.10, random_state=42):

    account_ids = df["account_id"].unique()
    val_size_proportion = val_size / (val_size + test_size)

    train_accounts, temp_accounts = train_test_split(account_ids, train_size=train_size, random_state=random_state, shuffle=True)
    val_accounts, test_accounts = train_test_split(temp_accounts, train_size=val_size_proportion, random_state=random_state,shuffle=True)

    train_df = df[df["account_id"].isin(train_accounts)].copy()
    val_df = df[df["account_id"].isin(val_accounts)].copy()
    test_df = df[df["account_id"].isin(test_accounts)].copy()

    return train_df, val_df, test_df


def validate_splits(df, train_df, val_df, test_df):
    # Check that no observations were lost
    assert (train_df.shape[0] + val_df.shape[0] + test_df.shape[0] == df.shape[0])

    # Check that accounts do not overlap across splits
    assert set(train_df["account_id"]).isdisjoint(val_df["account_id"])
    assert set(train_df["account_id"]).isdisjoint(test_df["account_id"])
    assert set(val_df["account_id"]).isdisjoint(test_df["account_id"])



def fit_transform_train(df):

    scaler = StandardScaler()
    transform_df = df.copy()
    
    features = transform_df[EMISSION_COLUMNS].copy()
    features["transaction_volume"] = np.log1p(features["transaction_volume"])
    scaled_features = scaler.fit_transform(features)
    transform_df["transaction_volume_log"] = scaled_features[:, 0]
    transform_df["cross_border_ratio_scaled"] = scaled_features[:, 1]
    transform_df["structuring_score_scaled"] = scaled_features[:, 2]

    return transform_df, scaler

def fit_transform_val_test(df, scaler):

    transform_df_val_test = df.copy()
    
    features = transform_df_val_test[EMISSION_COLUMNS].copy()
    features["transaction_volume"] = np.log1p(features["transaction_volume"])
    ### not using fit_transform since I am using the same scaler from "fit_transform_train" 
    scaled_features = scaler.transform(features) 
    transform_df_val_test["transaction_volume_log"] = scaled_features[:, 0]
    transform_df_val_test["cross_border_ratio_scaled"] = scaled_features[:, 1]
    transform_df_val_test["structuring_score_scaled"] = scaled_features[:, 2]

    return transform_df_val_test


def build_sequences(df):
    """
    Build one observation sequence per account.

    """
    sequences = []
    sequence_columns = ["transaction_volume_log", "cross_border_ratio_scaled","structuring_score_scaled"]

    df_sorted = df.sort_values(["account_id", "time"])

    for _, account_df in df_sorted.groupby("account_id", sort=True):
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
    
    validate_splits(df, train_df, val_df, test_df)

    train_df, scaler = fit_transform_train(train_df)

    val_df = fit_transform_val_test(val_df, scaler)
    test_df = fit_transform_val_test(test_df, scaler)

    train_sequences = build_sequences(train_df)
    val_sequences = build_sequences(val_df)
    test_sequences = build_sequences(test_df)

    return train_df, val_df,test_df, train_sequences, val_sequences, test_sequences, scaler
    