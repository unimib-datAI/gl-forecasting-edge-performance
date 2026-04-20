from utils.centralized_training import aggregate_datasets
from utils.data import save_dataset

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import json
import os

import datetime
from math import ceil
from sklearn.model_selection import train_test_split



# Function to import data
def import_data(path):
  df = pd.DataFrame()
  for node_type in node_types:
      # Retrieve all files in the data folder
      file_csv = [file for file in os.listdir(path + node_type) if file.endswith('.csv')]
      # Create the dataframe by concatenating all read files
      dataframes = []
      for file in file_csv:
          file_path = os.path.join(path + node_type, file)
          df_temp = pd.read_csv(file_path)
          # Remove the columns in the dataframe that begin with "function_"
          df_temp.drop(columns=[col for col in df_temp if col.startswith('function_')], inplace=True)
          # Add the column "node_type" and assign the value of 'type' to all rows
          if node_type == "HEAVY":
              df_temp["node_type"] = 0
          elif node_type == "MID":
              df_temp["node_type"] = 1
          else:
              df_temp["node_type"] = 2

          dataframes.append(df_temp)

      df = pd.concat([df, *dataframes], axis=0, ignore_index=True)

  return df


# Function used to fill NaN values within the dataframe X
def fill_NaN(X):
  for col in X:
    if(col.startswith('success_rate_')):
      X.loc[:, col] = X.loc[:, col].fillna(1)
    else:
      X.loc[:, col] = X.loc[:, col].fillna(0)
  return X

# Function to assign zero to noise metrics where requests rates are 0
def assign_zero_to_noise(df):
  functions = [col[14:] for col in df if col.startswith('rate')]

  for function in functions:
      df.loc[df['rate_function_' + function] == 0, ['cpu_usage_function_' + function, 'ram_usage_function_' + function, 'power_usage_function_' + function, 'replica_' + function]] = 0
  return df

# Function to select relevant columns
def select_columns(df):
  targets = [col for col in df if (col.startswith('cpu_usage_') or
                                   col.startswith('ram_usage_') or
                                   col.startswith('overloaded_node')
                                   or col.startswith('medium_latency')) and 'idle' not in col]# or col.startswith('replica')
  features = [col for col in df if col.startswith('rate_') or 'node_type' in col]
  features.sort()
  df = df[features+targets]
  return df

# Function to Rename Eat Memory column's Name
def rename_eat_memory(df):
  new_column_name = {i:i.replace("-","_") for i in df.keys() if "eat" in i}
  df.rename(columns=new_column_name, inplace=True)
  return df

# Function to add overload status and overload ratio.

# 0 = all 0, 1 = all 1, 2 = mix
def group_status(x):
    if (x == 0).all():
        return 0
    elif (x == 1).all():
        return 1
    else:
        return 2
    
def overload_status_ratio(df,features,target):
  grouped = df.groupby(features)

  df['overloaded_status'] = grouped[target].transform(group_status)

  # Ratio of 1’s in the group
  df['overloaded_ratio'] = grouped[target].transform('mean')

  return df

def get_node_capacity(node_id: int) -> int:
  # Lookup for node capacities
  GB = 1024 ** 3  # 1 GB in bytes
  capacities = {
      0: 24 * GB,   # Heavy
      1: 16 * GB,   # Mid
      2:  8 * GB    # Light
  }
  return capacities[node_id]

def ram_usage_to_percentage(ram_usage, node_type):
    """Return RAM utilization % given usage (bytes) and node type."""
    return (ram_usage / get_node_capacity(node_type)) * 100

# Function to compute theoretical RAM usage percentage.
def compute_ram_usage_percentage_theoretical(df, ram_col="ram_usage_node", node_col="node_type"):
    """Apply ram_usage_to_perc on the DataFrame."""
    # df["ram_usage_node_perc_theor"] = df.apply(
    #     lambda row: ram_usage_to_percentage(row[ram_col], row[node_col]),
    #     axis=1
    # )
    # Compute values
    perc_values = df.apply(
        lambda row: ram_usage_to_percentage(row[ram_col], row[node_col]),
        axis=1
    )

    # Find the index of ram_usage column
    insert_at = df.columns.get_loc("ram_usage_node_percentage") + 1

    # Insert new column at that position
    df.insert(insert_at, "ram_usage_node_perc_theor", perc_values)

    return df


#Remove rows where all specified columns have value 0.
def remove_baselines(df,features):
    df = df.copy()
    return df[~(df[features] == 0).all(axis=1)]

# Function to remove outliers
def remove_outliers(df):
  # Iterate over each target column and handle outliers
  functions_column = [col for col in df if col.startswith('rate')]
  targets = [col for col in df if (col.startswith('power_usage_') or col.startswith('cpu_usage_') or col.startswith('ram_usage_') or col.startswith('overloaded_node') or col.startswith('medium_latency')) and 'idle' not in col]
  grouped = df.groupby(functions_column + ['node_type'])
  threshold = 1
  for target in targets:
      print(target)
      if target != 'overloaded_node':
          mean = grouped[target].transform('mean')
          std = grouped[target].transform('std')
          outliers = (df[target] > mean + threshold * std) | (df[target] < mean - threshold * std)
          print(outliers.sum())
          df[target] = df[target].where(~outliers, mean)
      else:
          # Replace the overloaded value of the group by the mode.
          new_overloaded = grouped[target].transform(lambda x: x.mode().iloc[0])
          df['overloaded_node'] = new_overloaded
          print(df["overloaded_node"].value_counts())
  df_only_useful = df[functions_column + targets]
  return df

# creation of function to perform all the processes ---------------------------------------

# data preparation
def prepare_single_dataset(path_to_csvs: str, features: list) -> pd.DataFrame:
    df = import_data(path_to_csvs)
    df = fill_NaN(df)
    df = assign_zero_to_noise(df)
    df = select_columns(df)
    df = rename_eat_memory(df)
    df = overload_status_ratio(df, features, "overloaded_node")
    df = compute_ram_usage_percentage_theoretical(
        df,
        ram_col="ram_usage_node",
        node_col="node_type"
    )
    return df

def clean_for_model(df: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    df = remove_baselines(df, numerical_features)
    df = remove_outliers(df)
    return df
# build task and dataset dictionaries
def build_tasks_and_datasets(df: pd.DataFrame):
    target_prefixes = ("cpu_usage_", "ram_usage_", "overloaded_node", "medium_latency")
    targets = [
        col for col in df.columns
        if col.startswith(target_prefixes) and "idle" not in col
    ]

    categorical_features = ["node_type"]
    numerical_features = [col for col in df.columns if col.startswith("rate_")]
    features = categorical_features + numerical_features

    categorical_targets = ["overloaded_node"]
    numerical_targets = [col for col in targets if col not in categorical_targets]

    tasks = {
        "Multi_Task_regression": {
            "features": features,
            "targets": numerical_targets
        },
        "Multi_Task_classification": {
            "features": features,
            "targets": categorical_targets
        }
    }

    tasks_unified = {
        "Multi_Task": {
            "features": features,
            "targets": numerical_targets + categorical_targets,
            "regression_targets": numerical_targets,
            "classification_targets": categorical_targets
        }
    }

    return features, numerical_features, tasks, tasks_unified

def prepare_feature_target_datasets_cv(df, tasks):
    """
    Prepare datasets for multiple targets regression and overloaded node classification
    with optional SMOTENC oversampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset containing both features and targets.
    features : list of str
        List of feature column names.
    targets : list of str
        List of target column names.
    categorical_features : list of str
        Categorical feature columns to be passed to SMOTENC.

    Returns
    -------
    features_datasets : dict
        Mapping target names to feature DataFrames (oversampled if needed).
    target_datasets : dict
        Mapping target names to target Series.
    """

    feature_dataset = {}
    target_dataset = {}
    df_augmented = None
    for task_name, task_info in tasks.items():
      X = df[task_info["features"]]
      y = df[task_info["targets"]]
      feature_dataset[task_name] = X
      target_dataset[task_name] = y

    return feature_dataset, target_dataset

# final data preparation function
def prepare_source_target_datasets(path_source: str, path_target: str):
    initial_features = [
        "node_type",
        "rate_function_curl",
        "rate_function_eat_memory",
        "rate_function_env",
        "rate_function_figlet",
        "rate_function_nmap",
        "rate_function_shasum",
    ]

    df_source = prepare_single_dataset(path_source, initial_features)
    df_target = prepare_single_dataset(path_target, initial_features)

    features, numerical_features, tasks, tasks_unified = build_tasks_and_datasets(df_source)

    df_source = clean_for_model(df_source, numerical_features)
    df_target = clean_for_model(df_target, numerical_features)

    feature_dataset_source, target_dataset_source = prepare_feature_target_datasets_cv(
        df_source, tasks_unified
    )
    feature_dataset_target, target_dataset_target = prepare_feature_target_datasets_cv(
        df_target, tasks_unified
    )

    return {
        "df_source": df_source,
        "df_target": df_target,
        "features": features,
        "numerical_features": numerical_features,
        "tasks": tasks,
        "tasks_unified": tasks_unified,
        "feature_dataset_source": feature_dataset_source,
        "target_dataset_source": target_dataset_source,
        "feature_dataset_target": feature_dataset_target,
        "target_dataset_target": target_dataset_target,
    }

# training time dataframe ----------------------------------------------------


# plot functions -------------------------------------------------------------------



# main function for testing ----------------------------------------

# if __name__ == "__main__":
#     path_to_csvs = "../data/raw/source_domain/"
#     path_to_csvs_target = "../data/raw/target_domain/"
#     path_to_networks = "../data/networks"
#     base_output_folder = "../experiments"
#     node_types = ["LIGHT", "MID", "HEAVY"]
#     prepare_source_target_datasets(path_to_csvs, path_to_csvs_target)

if __name__ == "__main__":
    path_to_csvs = "../data/raw/source_domain/"
    path_to_csvs_target = "../data/raw/target_domain/"
    node_types = ["LIGHT", "MID", "HEAVY"]

    result = prepare_source_target_datasets(path_to_csvs, path_to_csvs_target)

    df_source = result["df_source"]

    # output_path = "/Users/kingsley/Documents/TESI/TEST_RESULTS/TESTS/df_source.csv"
    # df_source.to_csv(output_path, index=False)
    # print(f"Salvato in: {output_path}")

    # SORT 
    df_sorted = (
        df_source
        .sort_values(by=df_source.columns.tolist())
        .reset_index(drop=True)
        .head(100)  # opzionale ma utile per confronto
    )
    #OUTPUT DIRECTORY
    output_dir = "/Users/kingsley/Documents/TESI/TEST_RESULTS/TESTS"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "df_source_head100_sorted.csv")
    # SAVE FOR CONRONTATION
    df_sorted.to_csv(output_path, index=False)
    print(f"✅ Salvato file ordinato in: {output_path}")

# if __name__ == "__main__":
#   path_to_csvs = "../data/raw/source_domain/"
#   path_to_networks = "../data/networks"
#   base_output_folder = "../experiments"
#   node_types = ["LIGHT", "MID", "HEAVY"]
#   n = 10
#   k = 3
#   seed = 4850
#   simulations = range(10)
#   for simulation in simulations:
#     print(f"{20*'-'} {simulation} {20*'-'}")
#     prepare_data(
#       node_types, 
#       path_to_csvs, 
#       path_to_networks, 
#       base_output_folder, 
#       n, 
#       k, 
#       seed, 
#       simulation, 
#       0.1, 
#       0.1
#     )


# functions to display results --------------------------------------------------------------

from IPython.display import display, HTML
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

def display_results(results_df):
  for task_name, df in results_df.items():
    printmd(f"# {task_name}")
    printmd("---")
    display(HTML(df.to_html()))

# PREPROCESS_FOLD---------------------------------------------------------------------------------------------------------------
def check_node_type_distribution(X_train, X_val, X_test, node_col='node_type', normalize=True): # used in preprocessing(not used in kan) and preprocess_fold
    """
    Check and compare the distribution of `node_type` (or any categorical feature)
    across train, validation, and test splits.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        DataFrames for training, validation, and test sets.
    node_col : str, default='node_type'
        Name of the column representing node types (or any categorical variable).
    normalize : bool, default=True
        If True, show proportions (%) instead of raw counts.

    Returns
    -------
    pd.DataFrame
        Summary table comparing distributions across splits.
    """
    # Validate column existence
    for split_name, df in zip(['Train', 'Val', 'Test'], [X_train, X_val, X_test]):
        if node_col not in df.columns:
            raise ValueError(f"'{node_col}' not found in {split_name} set.")

    # Compute distributions
    def get_dist(df, name):
        return df[node_col].value_counts(normalize=normalize).rename(name)

    dist_train = get_dist(X_train, 'Train')
    dist_val   = get_dist(X_val, 'Val')
    dist_test  = get_dist(X_test, 'Test')

    # Combine into one comparison DataFrame
    dist_summary = pd.concat([dist_train, dist_val, dist_test], axis=1).fillna(0)
    if normalize:
        dist_summary = dist_summary * 100  # show as %

    # Add total rows for reference
    total_counts = {
        "Train Total": len(X_train),
        "Val Total": len(X_val),
        "Test Total": len(X_test)
    }
    totals_df = pd.DataFrame(total_counts, index=["Total Rows"])

    printmd("## Node Type Distribution Across Splits:")
    print(dist_summary.round(2))
    print("\n Dataset Sizes:")
    print(totals_df)

    return dist_summary



def check_data_leakage(X_train, X_val, X_test, group_cols=None): # used in preprocessing and preprocess_fold
    """
    Verify that there is no data leakage (i.e., no repeated feature combinations)
    across train, validation, and test splits.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        DataFrames for train, validation, and test sets.
    group_cols : list of str, optional
        Columns to consider for checking duplicates.
        If None, all columns of X_train are used.

    Returns
    -------
    None
        Prints leakage report. Raises AssertionError if leakage is detected.
    """


    if group_cols is None:
        group_cols = X_train.columns.tolist()

    # Create a unique signature for each feature combination
    def make_signatures(df):
        return set(df[group_cols].astype(str).agg('-'.join, axis=1))

    sig_train = make_signatures(X_train)
    sig_val   = make_signatures(X_val)
    sig_test  = make_signatures(X_test)

    # Check overlaps
    leak_train_val = sig_train.intersection(sig_val)
    leak_train_test = sig_train.intersection(sig_test)
    leak_val_test = sig_val.intersection(sig_test)

    total_leaks = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)

    printmd("## Data Leakage Check:")

    print(f"Train Samples: {len(X_train)}")
    print(f"Val Samples: {len(X_val)}")
    print(f"Test Samples: {len(X_test)}")

    print("\n")
    # print(f"- Train–Val overlaps: {len(leak_train_val)}")
    # print(f"- Train–Test overlaps: {len(leak_train_test)}")
    # print(f"- Val–Test overlaps: {len(leak_val_test)}")

    if total_leaks == 0:
        print("No data leakage detected. Splits are clean.")
    else:
        print("Data leakage detected!")
        if len(leak_train_val):
            print(f"  → {len(leak_train_val)} overlapping feature groups between Train and Val")
        if len(leak_train_test):
            print(f"  → {len(leak_train_test)} overlapping feature groups between Train and Test")
        if len(leak_val_test):
            print(f"  → {len(leak_val_test)} overlapping feature groups between Val and Test")
        #raise AssertionError("Data leakage detected between splits.")

def fit_minmax_scaler(X, y=None, regression_cols=None, classification_col=None): # used in preprocessing(not used in kan notebook) and preprocess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Fit MinMaxScaler for features (and optionally regression targets).

    X:
      - node_type column is excluded from scaling

    y:
      - only regression columns are scaled
      - classification column remains unchanged
    """

    # ----------------------
    # 1. Handle feature scaler
    # ----------------------
    scaler_x = MinMaxScaler(feature_range=(1, 2))
    # Drop node_type if exists
    if "node_type" in X.columns:
        scaler_x.fit(X.drop(columns=["node_type"]))
    else:
        scaler_x.fit(X)

    # ----------------------
    # 2. Handle target scaler
    # ----------------------
    scaler_y = None
    if y is not None and regression_cols is not None:
        # Fit scaler on regression columns only
        scaler_y = MinMaxScaler(feature_range=(1, 2))
        scaler_y.fit(y[regression_cols])

    # ---- Logging ----
    print("[INFO] MinMaxScaler fitted:")
    print(f"  - Features shape: {X.shape}")
    if y is not None:
        print(f"  - Regression targets shape: {y[regression_cols].shape}")
        print(f"  - Classification column '{classification_col}' excluded from scaling")
    print()

    return scaler_x, scaler_y


def save_scalers(scaler_x, scaler_y, task_name, base_path='./scalers/functions/'): # used in preprocessing(not used in kan notebook) and preprocess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Save fitted scalers for features and targets.

    Parameters
    ----------
    scaler_x : MinMaxScaler
        Fitted feature scaler.
    scaler_y : MinMaxScaler or None
        Fitted target scaler (if any).
    task_name : str
        Name of the task (e.g., 'Multi_Target_regression', 'overloaded_node_classification')
    base_path : str, default='./scalers/functions/'
        Base directory to save scalers.
    """
    # Create directories
    x_dir = os.path.join(base_path, 'scaler_x')
    y_dir = os.path.join(base_path, 'scaler_y')
    os.makedirs(x_dir, exist_ok=True)
    os.makedirs(y_dir, exist_ok=True)

    # Save X scaler
    joblib.dump(scaler_x, os.path.join(x_dir, f"{task_name}_x.joblib"))

    # Save Y scaler (only if provided)
    if scaler_y is not None:
        joblib.dump(scaler_y, os.path.join(y_dir, f"{task_name}_y.joblib"))

    print(f"[INFO] Scalers saved for target '{task_name}'")
    if scaler_y is None:
        print("  - No target scaler (classification or categorical target).")
    print()

def transform_with_scalers( # used in preprocessing and preprocess_fold 
    X_train, X_val, X_test, scaler_x,
    y_train=None, y_val=None, y_test=None,
    scaler_y=None, regression_cols=None, classification_col=None
):
    """
    Transform train/val/test using fitted scalers.

    X:
      - scales all columns except node_type
      - adds node_type back after scaling

    y:
      - scales only regression targets
      - keeps classification target unchanged
    """

    # =========================================
    # 1. Separate node_type from X
    # =========================================
    def split_node_type(df):
        if "node_type" in df.columns:
            node = df["node_type"].values.reshape(-1, 1)
            df_wo = df.drop(columns=["node_type"])
        else:
            node = None
            df_wo = df
        return df_wo, node


    X_train_noNT, node_train = split_node_type(X_train)
    X_val_noNT, node_val = split_node_type(X_val)
    X_test_noNT, node_test = split_node_type(X_test)

    # =========================================
    # 2. Scale numeric features only
    # =========================================
    X_train_scaled = scaler_x.transform(X_train_noNT)
    X_val_scaled   = scaler_x.transform(X_val_noNT)
    X_test_scaled  = scaler_x.transform(X_test_noNT)

    # Add back node_type (not scaled)
    if node_train is not None:
        X_train_scaled = np.hstack([node_train, X_train_scaled])
        X_val_scaled   = np.hstack([node_val, X_val_scaled ])
        X_test_scaled  = np.hstack([node_test, X_test_scaled])

    # =========================================
    # 3. Handle Y scaling
    # =========================================
    if y_train is not None and scaler_y is not None:

        # Extract and scale regression columns
        y_train_reg_s = scaler_y.transform(y_train[regression_cols])
        y_val_reg_s   = scaler_y.transform(y_val[regression_cols])
        y_test_reg_s  = scaler_y.transform(y_test[regression_cols])

        # Recombine regression + classification
        y_train_scaled = pd.DataFrame(y_train_reg_s, columns=regression_cols)
        y_val_scaled   = pd.DataFrame(y_val_reg_s, columns=regression_cols)
        y_test_scaled  = pd.DataFrame(y_test_reg_s, columns=regression_cols)

        # Add classification target unchanged
        y_train_scaled[classification_col] = y_train[classification_col].values
        y_val_scaled[classification_col]   = y_val[classification_col].values
        y_test_scaled[classification_col]  = y_test[classification_col].values

    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    # ---- Logging ----
    print("[INFO] Transformation complete.")
    print(f"  - X_train shape: {X_train_scaled.shape}")
    print(f"  - X_val shape: {X_val_scaled.shape}")
    print(f"  - X_test shape: {X_test_scaled.shape}\n")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled
    )

def transform(df, rate_columns=None, pad_value=0.0): # used in preprocessing and preprcess_fold and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Transform the input dataframe into padded, variable-length sequences.

    Each sample becomes a sequence of steps where each step = [rate, method_onehot(6), node_type].
    Steps with zero rate are removed. Then sequences are padded to the same length.

    Returns:
        X_padded: np.ndarray, shape (num_samples, max_seq_len, 8)
        mask: np.ndarray, shape (num_samples, max_seq_len), 1 for valid, 0 for padded
    """

    if rate_columns is None:
        rate_columns = [
            'rate_function_env', 'rate_function_curl', 'rate_function_eat_memory',
            'rate_function_nmap', 'rate_function_shasum', 'rate_function_figlet'
        ]

    node_type = df['node_type'].values
    num_samples = df.shape[0]
    sequence_length = len(rate_columns)  # 6

    # One-hot encode method indices (0–5)
    method_ids = np.arange(sequence_length).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    method_onehot = encoder.fit_transform(method_ids)  # (6, 6)

    X_sequences = []

    for i in range(num_samples):
        node_val = node_type[i]
        rate_values = df.loc[i, rate_columns].values  # shape (6,)

        # Keep only non-zero steps
        nonzero_mask = rate_values != 1
        if not np.any(nonzero_mask):
            nonzero_mask = np.array([True])  # Keep one dummy step if all zero

        rate_seq = rate_values[nonzero_mask].reshape(-1, 1)
        method_seq = method_onehot[nonzero_mask]
        node_seq = np.full((np.sum(nonzero_mask), 1), node_val)

        seq = np.concatenate([rate_seq, method_seq, node_seq], axis=1)  # (L_i, 8)
        X_sequences.append(seq)

    # Pad all sequences to same length (post-padding with zeros)
    max_len = max(len(seq) for seq in X_sequences)
    X_padded = pad_sequences(X_sequences, maxlen=max_len, dtype='float32', padding='post', value=pad_value)

    # Create mask (1 for real steps, 0 for padded)
    mask = np.array([[1]*len(seq) + [0]*(max_len - len(seq)) for seq in X_sequences], dtype='float32')

    # Drop node_type column if you want to reuse df
    df = df.drop(columns=['node_type'])

    return X_padded

def preprocess_fold(X_train, X_val, X_test, y_train, y_val, y_test,tasks,base_path): # used in cross_validate
    """
    Perform full preprocessing pipeline:
    - Split data into train/validation/test sets
    - Fit MinMax scalers for features and (optionally) targets
    - Save scalers to disk
    - Transform datasets using fitted scalers
    - Return scaled datasets and scalers for all tasks
    """
    printmd("## Preprocessing")
    # -----------------------------------------------------
    # Initialize dictionaries to store splits and scalers
    # -----------------------------------------------------
    x_train_dict, x_val_dict, x_test_dict = {}, {}, {}
    y_train_dict, y_val_dict, y_test_dict = {}, {}, {}
    x_scalers, y_scalers = {}, {}

    # -----------------------------------------------------
    # Loop through each task for preprocessing
    # -----------------------------------------------------
    for task_name,task_info in tasks.items():
        printmd(f"# {task_name}")
        printmd("---")


        # -------------------------------------------------
        # Check for data leakage
        # -------------------------------------------------
        check_node_type_distribution(X_train,X_val,X_test)
        check_data_leakage(X_train,X_val,X_test)

        # -------------------------------------------------
        # Fit MinMaxScaler
        # Skip target scaling for classification tasks
        # -------------------------------------------------
        printmd("## Scalaing")
        if task_name == "overloaded_node_classification":
            scaler_x, scaler_y = fit_minmax_scaler(X_train)
        else:
            scaler_x, scaler_y = fit_minmax_scaler(X_train, y_train,
                                                   regression_cols=task_info["regression_targets"],
                                                  classification_col=task_info["classification_targets"])

        # -------------------------------------------------
        # Save fitted scalers to disk
        # -------------------------------------------------
        save_scalers(scaler_x, scaler_y, task_name, base_path)

        # -------------------------------------------------
        # Apply scaling to train/val/test sets
        # -------------------------------------------------
        (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled
        ) = transform_with_scalers(
            X_train, X_val, X_test,
            scaler_x,
            y_train, y_val, y_test,
            scaler_y,
            regression_cols=task_info["regression_targets"],
            classification_col=task_info["classification_targets"]
        )


        # -------------------------------------------------
        # Optional: Apply any custom transformation on scaled features
        # Skip this step for classification tasks
        # -------------------------------------------------
        print(f"Transforming for Janoosy: {datetime.datetime.now()}")
        if task_name != "overloaded_node_classification":
            X_train_scaled = transform(pd.DataFrame(X_train_scaled, columns=features))
            X_val_scaled = transform(pd.DataFrame(X_val_scaled, columns=features))
            X_test_scaled = transform(pd.DataFrame(X_test_scaled, columns=features))

        # -------------------------------------------------
        # Store scaled datasets and scalers in dictionaries
        # -------------------------------------------------
        x_train_dict[task_name] = X_train_scaled
        x_val_dict[task_name] = X_val_scaled
        x_test_dict[task_name] = X_test_scaled
        x_scalers[task_name] = scaler_x

        y_train_dict[task_name] = y_train_scaled
        y_val_dict[task_name] = y_val_scaled
        y_test_dict[task_name] = y_test_scaled
        y_scalers[task_name] = scaler_y

    # -----------------------------------------------------
    # Return all processed splits and scalers
    # -----------------------------------------------------
    return (
        x_train_dict, x_val_dict, x_test_dict, x_scalers,
        y_train_dict, y_val_dict, y_test_dict, y_scalers
    )


# perform_custom_oversampling (used in cross_validate) ----------------------------------------------------------------------------------------------------------
def limit_identical_combinations(df, features, target_cols, max_per_combination=1): # used in perform_custom_oversampling
    """
    Limit the dataset to at most `max_per_combination` identical feature combinations.

    Parameters
    ----------
    df : pd.DataFrame
    features : list
        Columns to group by when identifying identical combinations.
    target_cols : list
        Target column names.
    max_per_combination : int
        Maximum allowed identical combinations per group.

    Returns
    -------
    df_limited : pd.DataFrame
        Reduced datasets after limiting duplicates.
    """
    print(f"\n=== Limiting to max {max_per_combination} identical combinations ===")

    # df_combined = pd.concat([X, y], axis=1)
    df_limited = (
        df.groupby(features, as_index=False)
        .head(max_per_combination)
        .reset_index(drop=True)
    )
# old version
    #df_limited = (
        #df.groupby(features, group_keys=False)
        #.apply(lambda x: x.head(max_per_combination))
        #.reset_index(drop=True)
    #)

    X_limited = df_limited[features]
    y_limited = df_limited[target_cols]

    print(f"Original row count: {len(df)}")
    print(f"After limiting: {len(df_limited)} (removed {len(df) - len(df_limited)})")
    print("\nClass distribution after limiting:")
    print(y_limited.value_counts())

    return df_limited


def generate_synthetic_overloaded(df, features, target_col): # used in perform_custom_oversampling
    """
    Generate synthetic overloaded samples to balance classes per node_type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (after limiting).
    features : list
        Feature column names.
    target_col : str
        Binary target column name (0 = non-overloaded, 1 = overloaded).

    Returns
    -------
    df_augmented : pd.DataFrame
        Dataset containing newly generated synthetic rows.
    """
    print("\n=== Generating synthetic overloaded samples (Unique Features Balancing) ===")

    rate_cols = [col for col in df.columns if col.startswith("rate_function_")]
    node_types = df["node_type"].unique()
    augmented_rows = []
    generated_total = 0

    existing_keys = set(tuple(row[col] for col in features) for _, row in df.iterrows())

    for nt in node_types:
        subset = df[df["node_type"] == nt]
        count_1 = subset[subset[target_col] == 1].shape[0]
        count_0 = subset[subset[target_col] == 0].shape[0]
        needed = count_0 - count_1

        print(f"\n[Node Type {nt}] Overloaded: {count_1}, Non-overloaded: {count_0}")
        if needed <= 0:
            print(f"Already balanced or overloaded.")
            continue

        print(f"Generating {needed} synthetic samples for node_type {nt}")
        df_overloaded = subset[subset[target_col] == 1]
        generated = 0

        for _, row in df_overloaded.iterrows():
            overloaded_funcs = [col for col in rate_cols if row[col] > 0]
            if len(overloaded_funcs) != 3:
                continue

            fixed_rates = {col: row[col] for col in overloaded_funcs}

            for target_func in overloaded_funcs:
                start = int(fixed_rates[target_func])
                for val in range(start + 5, 201, 10):
                    if generated >= needed:
                        break

                    new_row = {col: 0.0 for col in rate_cols}
                    for col in overloaded_funcs:
                        new_row[col] = val if col == target_func else fixed_rates[col]

                    # Fill metadata and targets
                    new_row["node_type"] = nt
                    new_row[target_col] = 1

                    # Add other columns from original
                    for col in df.columns:
                        if col not in new_row and col not in rate_cols:
                            new_row[col] = row[col]

                    new_key = tuple(new_row[col] for col in features)
                    if new_key not in existing_keys:
                        augmented_rows.append(new_row)
                        existing_keys.add(new_key)
                        generated += 1
                        generated_total += 1

                if generated >= needed:
                    break
            if generated >= needed:
                break

        print(f"Generated {generated} samples for node_type {nt}")

    print(f"\n Total synthetic samples generated: {generated_total}")
    return pd.DataFrame(augmented_rows)


def balance_df_nodewise(df, df_augmented, node_col='node_type', target_col='overloaded_node'): # used in perform_custom_oversampling
    """
    Balance df within each node_type based on overloaded_node (target).
    df_augmented contains only rows with overloaded_node = 1.
    Includes detailed debug prints.
    """

    balanced_parts = []
    node_types = df[node_col].unique()

    print("\n========= ROWS BALANCING =========\n")
    print(f"Node types found: {list(node_types)}\n")

    for nt in node_types:
        print(f"\n--- Processing node_type: {nt} ---")

        subset = df[df[node_col] == nt].copy()

        # Count original samples
        count_1 = (subset[target_col] == 1).sum()
        count_0 = (subset[target_col] == 0).sum()

        print(f"Original counts → overloaded=0: {count_0}, overloaded=1: {count_1}")

        # Check if balance is needed
        if count_1 >= count_0:
            print("Already balanced or overloaded=1 is majority. No augmentation needed.")
            balanced_parts.append(subset)
            continue

        needed = count_0 - count_1
        print(f"Need to add {needed} more rows for overloaded=1")

        # Filter augmented rows for this node_type
        aug_rows = df_augmented[df_augmented[node_col] == nt]

        if aug_rows.empty:
            print(f"[WARNING] df_augmented has NO rows for node_type={nt}. Cannot balance this group!")
            balanced_parts.append(subset)
            continue

        print(f"Augmented rows available for this node_type={nt}: {len(aug_rows)}")

        # Calculate repeat factor
        repeats = ceil(needed / len(aug_rows))
        print(f"Repeating augmented rows {repeats} times")

        repeated = pd.concat([aug_rows] * repeats, ignore_index=True)

        rows_to_add = repeated.iloc[:needed]
        print(f"Rows actually added: {len(rows_to_add)}")

        # Append augmented rows
        subset = pd.concat([subset, rows_to_add], ignore_index=True)

        # Verify new counts
        new_count_1 = (subset[target_col] == 1).sum()
        new_count_0 = (subset[target_col] == 0).sum()
        print(f"After augmentation → overloaded=0: {new_count_0}, overloaded=1: {new_count_1}")

        balanced_parts.append(subset)

    print("\n========= BALANCING COMPLETED =========\n")
    final_df = pd.concat(balanced_parts, ignore_index=True)
    return final_df


def perform_custom_oversampling(df, features, reg_target, class_target): # used in prepare_feature_target_datasets(not in the notebook) and cross_validate
    """
    Custom domain-aware oversampling replacing SMOTENC.

    Steps:
    1. Limit identical feature combinations.
    2. Generate synthetic overloaded samples per node_type to balance unique features(Group level balance).
    3. Row-level Balance by duplicating augmented rows and returning a balanced dataset.

    Parameters
    ----------
    df : pd.DataFrame
    features : list
        Feature columns used for grouping and synthesis.
    reg_target : list
        Regression Target column(s).
    class_target : list
        Classification Target column(s).

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.DataFrame
        Balanced feature and target datasets.
    """
    printmd("## Oversampling")
    # Step 1: Apply limiting rule
    df_class = df[features + class_target]
    df_limited = limit_identical_combinations(df_class, features, class_target)

    # Step 2: Combine for rebalancing
    # df_limited = pd.concat([X_limited, y_limited], axis=1)
    target_col = class_target[0]

    # Step 3: Generate new samples to balance limited dataset
    df_augmented = generate_synthetic_overloaded(df_limited, features, target_col)

    # Optionally save the augmented rows
    os.makedirs(PATH_TO_AUGMENTED_ROWS, exist_ok=True)
    df_augmented.to_csv(PATH_TO_AUGMENTED_ROWS + "augmented_overloaded_samples.csv", index=False)
    print("Augmented rows saved to 'augmented_overloaded_samples.csv'")

    # Step 4: Final overall balance dataset (repeated features)
    df_balanced = balance_df_nodewise(df, df_augmented)
    X_res = df_balanced[features]
    y_res = df_balanced[reg_target + class_target]

    print("\nFinal class distribution after rebalancing:")
    print(y_res[class_target].value_counts())
    print("\nFinal class distribution after rebalancing (node_type-wise):")
    print(df_balanced.groupby("node_type")["overloaded_node"].value_counts().unstack(fill_value=0))

    print("\nFinal node_type distribution:")
    print(X_res["node_type"].value_counts())

    print("\n========= FINAL UNIQUE FEATURES SUMMARY =========")

    # Convert to tuples for unique combinations
    all_unique = df_balanced[features].drop_duplicates()
    unique_0 = df_balanced[df_balanced[target_col] == 0][features].drop_duplicates()
    unique_1 = df_balanced[df_balanced[target_col] == 1][features].drop_duplicates()

    print(f"Total unique feature combinations       : {len(all_unique)}")
    print(f"Unique feature combinations for 0       : {len(unique_0)}")
    print(f"Unique feature combinations for 1       : {len(unique_1)}")

    print("==========================================\n")

    return X_res, y_res, df_augmented, df_balanced



# NOT USED IN KAN NOTEBOOK -------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Remove rows which are overloaded (overloaded_node == 1).
def remove_overloaded_rows(df): # never used in kan 
    df = df.copy()
    return df[df["overloaded_node"] == 0]

def check_node_type_distribution_cv(X_train, X_val, node_col='node_type', normalize=True): # the cv version was used in kan notebook that doesn't control the dsitribution in test
    """
    Check distribution of node_type in CV Train and Val splits.
    """
    # Validate column existence
    for split_name, df in zip(['Train', 'Val'], [X_train, X_val]):
        if node_col not in df.columns:
            raise ValueError(f"'{node_col}' not found in {split_name} set.")

    # Compute distribution
    def get_dist(df, name):
        return df[node_col].value_counts(normalize=normalize).rename(name)

    dist_train = get_dist(X_train, 'Train')
    dist_val   = get_dist(X_val, 'Val')

    # Combine
    dist_summary = pd.concat([dist_train, dist_val], axis=1).fillna(0)
    if normalize:
        dist_summary = dist_summary * 100

    # Total sizes
    totals_df = pd.DataFrame({
        "Train Total": [len(X_train)],
        "Val Total": [len(X_val)]
    }, index=["Total Rows"])

    print("## Node Type Distribution Across CV Splits:")
    print(dist_summary.round(2))
    print("\n Dataset Sizes:")
    print(totals_df)

    return dist_summary

def check_data_leakage_cv(X_train, X_val, group_cols=None): # never used in kan notebook, the not cv version is used the only difference is that it doesn't use test
    """
    Check if any feature group appears in both Train and Val.
    """
    if group_cols is None:
        group_cols = X_train.columns.tolist()

    # Create signatures per row to detect duplicates
    def make_signatures(df):
        return set(df[group_cols].astype(str).agg('-'.join, axis=1))

    sig_train = make_signatures(X_train)
    sig_val   = make_signatures(X_val)

    # Check leakage
    leakage = sig_train.intersection(sig_val)

    print("## Data Leakage Check (CV):")
    print(f"Train Samples: {len(X_train)}")
    print(f"Val Samples:   {len(X_val)}")

    if len(leakage) == 0:
        print("\n No leakage — Train and Val are fully separated.\n")
    else:
        print(f"\n Data leakage detected: {len(leakage)} duplicate feature signatures shared between Train & Val!\n")
        # You may want to raise instead:
        # raise AssertionError("Data leakage detected between Train and Val splits.")

def perform_oversampling(X, y, categorical_features, target_name=None, random_state=42): # not used in kan notebook, but became a comment for prepare_feature_target_datasets
    """
    Apply SMOTENC oversampling to handle class imbalance in datasets
    containing categorical features. Prints before/after class distributions.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature set.
    y : pandas.Series
        Target variable.
    categorical_features : list of str
        Names of categorical columns in X.
    target_name : str, optional
        Name of the target for display/logging.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_resampled, y_resampled : pandas.DataFrame, pandas.Series
        Oversampled feature and target datasets.
    """
    if target_name:
        print(f"\nPerforming SMOTENC for target: {target_name}")
    print("Class distribution before SMOTE:")
    print(y.value_counts())

    cat_indices = [X.columns.get_loc(col) for col in categorical_features]
    smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)

    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Class distribution after SMOTE:")
        print(y_resampled.value_counts())
        print("Categorical feature distribution after SMOTE:")
        for col in categorical_features:
            print(f"\n{col}:\n{X_resampled[col].value_counts()}")
    except Exception as e:
        print(f"Could not perform SMOTE: {e}")
        return X, y

    return X_resampled, y_resampled

def prepare_feature_target_datasets(df, tasks): # not used in kan notebook, the cv version is used instead that doesn't do oversampling
    """
    Prepare datasets for multiple targets regression and overloaded node classification
    with optional SMOTENC oversampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset containing both features and targets.
    features : list of str
        List of feature column names.
    targets : list of str
        List of target column names.
    categorical_features : list of str
        Categorical feature columns to be passed to SMOTENC.

    Returns
    -------
    features_datasets : dict
        Mapping target names to feature DataFrames (oversampled if needed).
    target_datasets : dict
        Mapping target names to target Series.
    """

    feature_dataset = {}
    target_dataset = {}
    df_augmented = None
    for task_name, task_info in tasks.items():
      # X = df[task_info["features"]]
      # y = df[task_info["targets"]]
      if "Multi_Task" in task_name:
        printmd(f"##{task_name}")
        # X, y = perform_oversampling( X, y, categorical_features, target_name=task_name)
        X, y, df_augmented, df_balanced = perform_custom_oversampling(df, task_info["features"], task_info["regression_targets"],
                                                         task_info["classification_targets"])
      feature_dataset[task_name] = X
      target_dataset[task_name] = y


    return feature_dataset, target_dataset, df_augmented, df_balanced

def transform_with_scalers_cv( # not used in kan notebook, but the not cv version is used instead
    X_train, X_val, scaler_x,
    y_train=None, y_val=None,
    scaler_y=None, regression_cols=None, classification_col=None
):
    """
    Transform train/val using fitted scalers for Cross-Validation.

    X:
      - scales all numeric columns except node_type
      - node_type is added back unchanged

    y:
      - scales only regression targets
      - classification column remains unchanged
    """

    # =========================================
    # 1. Separate node_type from X
    # =========================================
    def split_node_type(df):
        if "node_type" in df.columns:
            node = df["node_type"].values.reshape(-1, 1)
            df_wo = df.drop(columns=["node_type"])
        else:
            node = None
            df_wo = df
        return df_wo, node

    X_train_noNT, node_train = split_node_type(X_train)
    X_val_noNT, node_val     = split_node_type(X_val)

    # =========================================
    # 2. Scale numeric features
    # =========================================
    X_train_scaled = scaler_x.transform(X_train_noNT)
    X_val_scaled   = scaler_x.transform(X_val_noNT)

    # Re-add node_type unchanged
    if node_train is not None:
        X_train_scaled = np.hstack([node_train, X_train_scaled])
        X_val_scaled   = np.hstack([node_val,   X_val_scaled])

    # =========================================
    # 3. Scale y if target scaling enabled
    # =========================================
    if y_train is not None and scaler_y is not None:
        # Scale regression outputs
        y_train_reg_s = scaler_y.transform(y_train[regression_cols])
        y_val_reg_s   = scaler_y.transform(y_val[regression_cols])

        # Recreate DataFrames
        y_train_scaled = pd.DataFrame(y_train_reg_s, columns=regression_cols)
        y_val_scaled   = pd.DataFrame(y_val_reg_s, columns=regression_cols)

        # Append classification unchanged
        y_train_scaled[classification_col] = y_train[classification_col].values
        y_val_scaled[classification_col]   = y_val[classification_col].values
    else:
        y_train_scaled, y_val_scaled = y_train, y_val

    # ---- Logging ----
    print("[INFO] CV transformation complete.")
    print(f"  - X_train: {X_train_scaled.shape}")
    print(f"  - X_val:   {X_val_scaled.shape}\n")

    return (
        X_train_scaled, X_val_scaled,
        y_train_scaled, y_val_scaled
    )

def split_train_test(X, y, test_size=0.2, random_state=42, stratify=None): # not used in kan notebook
    """
    Perform a simple train–test split.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature set.
    y : pandas.Series or numpy.ndarray
        Target variable.
    test_size : float, default=0.2
        Proportion of dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion (useful for classification).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split datasets for training and testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return X_train, X_test, y_train, y_test

def split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42, stratify=None): # not used in kan notebook
    """
    Perform a train–validation–test split.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature set.
    y : pandas.Series or numpy.ndarray
        Target variable.
    val_size : float, default=0.1
        Proportion of dataset for validation (from the remaining after test split).
    test_size : float, default=0.2
        Proportion of dataset for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion (useful for classification).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
        Split datasets for training, validation, and testing.
    """
    # First, split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Adjust validation size relative to remaining data
    val_relative_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_temp if stratify is not None else None
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def split_test_nodewise(X_test_dict, Y_test_dict, tasks): # not used in kan notebook,became a comment might ha been used in cross validate to perform evaluation
    """
    Split test data into nodewise subsets for both regression and classification tasks.

    Handles 3D regression input (seq_len, feature_dim) and 2D classification input.

    Returns
    -------
    X_test_nodewise : dict
        {'regression': {node_type: X_subset, ...}, 'classification': {...}}
    Y_test_nodewise : dict
        {'regression': {node_type: y_subset, ...}, 'classification': {...}}
    """
    node_types = [0.0, 1.0, 2.0]
    X_test_nodewise = {}
    Y_test_nodewise = {}

    for node in node_types:
      X_test_nodewise[node] = {}
      Y_test_nodewise[node] = {}

    for task_name in tasks:

      if task_name.startswith('overloaded'):
        # --- Classification (2D input) ---
        X_cls, y_cls = X_test_dict[task_name], Y_test_dict[task_name]
        # node_type = first element of each row
        node_col_cls = X_cls[:, 0]
        for node in node_types:
            mask = node_col_cls == node
            X_test_nodewise[node][task_name] = X_cls[mask]
            Y_test_nodewise[node][task_name] = y_cls[mask]

      else:
        # --- Regression (3D input) ---
        X_reg, y_reg = X_test_dict[task_name], Y_test_dict[task_name]
        # node_type = last element of last time step for each sample
        node_col_reg = np.array([x[-1, -1] for x in X_reg])
        for node in node_types:
            mask = node_col_reg == node
            X_test_nodewise[node][task_name] = X_reg[mask]
            Y_test_nodewise[node][task_name] = y_reg[mask]


    return X_test_nodewise, Y_test_nodewise

def split_train_val_test_grouped(X, y, val_size=0.1, test_size=0.2,
                                 random_state=42, stratify=None, group_cols=None): # used in preprocessing (not in kan notebook)
    """
    Perform a train–validation–test split while ensuring no feature-level data leakage
    (i.e., identical feature rows stay in the same split).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or pd.DataFrame
        Target variable(s).
    val_size : float, default=0.1
        Validation set proportion (relative to remaining after test split).
    test_size : float, default=0.2
        Test set proportion of full dataset.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : array-like, default=None
        Optional stratification column/array.
    group_cols : list of str, optional
        Columns to define "uniqueness" of feature combinations.
        If None, all columns of X are used.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
        Group-consistent splits (no duplicate feature combinations across sets).
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame for group-based split.")

    if group_cols is None:
        group_cols = X.columns.tolist()

    # Step 1: Identify unique feature combinations
    X_unique = X[group_cols].drop_duplicates().reset_index(drop=True)

    # Step 2: Map each unique combination to its group index
    X['__group_id__'] = X[group_cols].astype(str).agg('-'.join, axis=1)
    group_ids = X_unique.astype(str).agg('-'.join, axis=1)

    # Step 3: Split group IDs (ensuring no overlap)
    group_train, group_test = train_test_split(
        group_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=None  # stratify not possible directly on groups
    )

    # Adjust validation size relative to train
    val_relative_size = val_size / (1 - test_size)
    group_train, group_val = train_test_split(
        group_train,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=None
    )

    # Step 4: Assign each sample based on group membership
    def mask_from_groups(groups):
        return X['__group_id__'].isin(groups)

    train_mask = mask_from_groups(group_train)
    val_mask = mask_from_groups(group_val)
    test_mask = mask_from_groups(group_test)

    # Step 5: Create splits
    X_train, X_val, X_test = X[train_mask].drop(columns='__group_id__'), X[val_mask].drop(columns='__group_id__'), X[test_mask].drop(columns='__group_id__')
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    # Final checks
    print(f"\nSplit Summary (group-consistent):")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    print(f"Unique feature groups: {len(group_ids)} (train={len(group_train)}, val={len(group_val)}, test={len(group_test)})")

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_with_scalers_crossdom( # used in prepare_cross_domain_evaluation_data(not used in the notebook)
    X, scaler_x,
    y=None, scaler_y=None,
    regression_cols=None,
    classification_col=None
):
    """
    Transform full feature and target datasets using fitted scalers
    for cross-domain evaluation (no train/val/test split).

    X:
      - scales all columns except node_type
      - adds node_type back as first column

    y:
      - scales only regression targets
      - keeps classification target unchanged
    """

    # =========================================
    # 1. Separate node_type from X
    # =========================================
    def split_node_type(df):
        if "node_type" in df.columns:
            node = df["node_type"].values.reshape(-1, 1)
            df_wo = df.drop(columns=["node_type"])
        else:
            node = None
            df_wo = df
        return df_wo, node

    X_noNT, node_vals = split_node_type(X)


    # =========================================
    # 2. Scale numeric features only
    # =========================================
    X_scaled_core = scaler_x.transform(X_noNT)

    # Add back node_type at the front
    if node_vals is not None:
        X_scaled = np.hstack([node_vals, X_scaled_core])
    else:
        X_scaled = X_scaled_core

    # =========================================
    # 3. Handle target scaling
    # =========================================
    if y is not None and scaler_y is not None:

        # Extract regression and classification
        y_reg_s = scaler_y.transform(y[regression_cols])

        # Reconstruct DataFrame
        y_scaled_df = pd.DataFrame(y_reg_s, columns=regression_cols)

        # Add classification as-is
        y_scaled_df[classification_col] = y[classification_col].values

        y_scaled = y_scaled_df

    else:
        y_scaled = y

    # =========================================
    # 4. Logging
    # =========================================
    print("[INFO] Cross-domain scaling complete.")
    print(f"  - X_scaled shape: {X_scaled.shape}")
    if y is not None:
        print(f"  - y_scaled shape: {np.array(y_scaled).shape}")

    return X_scaled, y_scaled

def prepare_cross_domain_evaluation_data(feature_dataset, target_dataset, tasks, base_path, augmented_df): # not used in kan notebook
    """
    Prepare datasets for cross-domain model evaluation.

    This function preprocesses feature and target data from both domains
    without splitting into train/validation/test sets. The entire dataset
    is scaled and returned for evaluation purposes (e.g., evaluating a
    source model on a target domain, and vice versa).

    Steps:
    - Fit MinMax scalers on the entire dataset (per task)
    - Skip target scaling for classification tasks
    - Save fitted scalers to disk
    - Transform full datasets using fitted scalers
    - Return scaled datasets and scalers for all tasks

    Parameters
    ----------
    feature_dataset : dict
        Dictionary of feature DataFrames for each task.
    target_dataset : dict
        Dictionary of target DataFrames/Series for each task.
    tasks : list
        List of task names to preprocess.
    base_path : str
        Directory path to save fitted scalers.

    Returns
    -------
    x_crossdom_dict : dict
        Dictionary of scaled feature DataFrames for each task.
    y_crossdom_dict : dict
        Dictionary of scaled target arrays for each task.
    x_crossdom_scalers : dict
        Dictionary of fitted feature scalers for each task.
    y_crossdom_scalers : dict
        Dictionary of fitted target scalers for each task.
    """

    # -----------------------------------------------------
    # Initialize dictionaries to store processed data
    # -----------------------------------------------------
    x_crossdom_dict, y_crossdom_dict = {}, {}
    x_crossdom_scalers, y_crossdom_scalers = {}, {}

    # -----------------------------------------------------
    # Loop through each task for preprocessing
    # -----------------------------------------------------
    for task_name, task_info in tasks.items():
        printmd(f"# {task_name}")
        printmd("---")
        # Extract feature and target datasets
        X = feature_dataset[task_name]
        y = target_dataset[task_name]

        X, y = remove_augmented_rows_from_test(X, y, augmented_df, features)
        # -------------------------------------------------
        # Fit MinMaxScaler
        # Skip target scaling for classification tasks
        # -------------------------------------------------
        if task_name == "overloaded_node_classification":
            scaler_x, scaler_y = fit_minmax_scaler(X)
        else:
            scaler_x, scaler_y = fit_minmax_scaler(X, y,
                                                   regression_cols=task_info["regression_targets"],
                                                  classification_col=task_info["classification_targets"])

        # -------------------------------------------------
        # Save fitted scalers to disk
        # -------------------------------------------------
        save_scalers(scaler_x, scaler_y, task_name, base_path)

        # -------------------------------------------------
        # Apply scaling to the full dataset
        # -------------------------------------------------
        X_scaled, y_scaled = transform_with_scalers_crossdom(
            X, scaler_x, y, scaler_y,
            regression_cols=task_info["regression_targets"],
            classification_col=task_info["classification_targets"]
        )


        # -------------------------------------------------
        # Optional: Apply any custom transformation
        # Skip this step for classification tasks
        # -------------------------------------------------
        # if task_name != "overloaded_node_classification":
        X_scaled = transform(pd.DataFrame(X_scaled, columns=features))

        # -------------------------------------------------
        # Store scaled data and scalers
        # -------------------------------------------------
        x_crossdom_dict[task_name] = X_scaled
        y_crossdom_dict[task_name] = y_scaled
        x_crossdom_scalers[task_name] = scaler_x
        y_crossdom_scalers[task_name] = scaler_y

    # -----------------------------------------------------
    # Return all processed datasets and scalers
    # -----------------------------------------------------
    return (
        x_crossdom_dict,
        y_crossdom_dict,
        x_crossdom_scalers,
        y_crossdom_scalers
    )

def remove_augmented_rows_from_test(x_test, y_test, augmented_df, features, verbose=True): # used in preprocessing(not used in kan) and prepare_cross_domain_evaluation_data(not used in the notebook)
    """
    Remove rows from (x_test, y_test) where feature combinations already exist in augmented_df.

    Parameters
    ----------
    x_test : pd.DataFrame
        Test feature set.
    y_test : pd.DataFrame or pd.Series
        Test target set.
    augmented_df : pd.DataFrame
        DataFrame containing augmented samples (with same feature columns).
    features : list
        List of feature column names to use for matching.
    verbose : bool, default=True
        Whether to print debug information and stats.

    Returns
    -------
    x_test_filtered : pd.DataFrame
        Test features after removing overlapping rows.
    y_test_filtered : pd.DataFrame or pd.Series
        Corresponding test targets after filtering.
    """

    # --- Safety checks ---
    missing_in_test = [f for f in features if f not in x_test.columns]
    missing_in_aug = [f for f in features if f not in augmented_df.columns]
    if missing_in_test or missing_in_aug:
        raise ValueError(f"Missing columns — In test: {missing_in_test}, In augmented: {missing_in_aug}")

    if verbose:
        print(f"\nMatching test samples against augmented_df on {len(features)} features: {features}")

    # --- Drop duplicates from augmented features to speed up merge ---
    aug_unique = augmented_df[features].drop_duplicates()
    if verbose:
        print(f"Unique combinations in augmented_df: {len(aug_unique)} (from {len(augmented_df)})")

    # --- Identify overlapping rows ---
    merged = x_test.merge(aug_unique, on=features, how='left', indicator=True)

    merged.index = x_test.index  # re-align indices to match x_test

    overlaps_mask = merged['_merge'] == 'both'
    removed_mask = merged['_merge'] == 'left_only'

    n_total = len(x_test)
    n_overlaps = overlaps_mask.sum()
    n_remaining = removed_mask.sum()


    # --- Filter test set ---
    x_test_filtered = x_test[removed_mask].reset_index(drop=True)

    y_test_filtered = y_test.loc[removed_mask.values].reset_index(drop=True)

    unique_combos = x_test_filtered[features].drop_duplicates().shape[0]

    # --- Debug stats ---
    if verbose:
        print("\nTest Filter Summary:")
        print(f"   Total test samples      : {n_total}")
        print(f"   Overlaps with augmented  : {n_overlaps}")
        print(f"   Remaining test samples   : {n_remaining}")
        print(f"   Rows removed percentage  : {n_overlaps / n_total * 100:.2f}%")
        print(f"   Unique feature combinations    : {unique_combos}")

    return x_test_filtered, y_test_filtered

def preprocessing(feature_dataset, target_dataset, tasks, base_path, augmented_df): # not used i kan notebook
    """
    Perform full preprocessing pipeline:
    - Split data into train/validation/test sets
    - Fit MinMax scalers for features and (optionally) targets
    - Save scalers to disk
    - Transform datasets using fitted scalers
    - Return scaled datasets and scalers for all tasks
    """
    printmd("## Preprocessing")
    # -----------------------------------------------------
    # Initialize dictionaries to store splits and scalers
    # -----------------------------------------------------
    x_train_dict, x_val_dict, x_test_dict = {}, {}, {}
    y_train_dict, y_val_dict, y_test_dict = {}, {}, {}
    x_scalers, y_scalers = {}, {}

    # -----------------------------------------------------
    # Loop through each task for preprocessing
    # -----------------------------------------------------
    for task_name,task_info in tasks.items():
        printmd(f"# {task_name}")
        printmd("---")
        # Extract feature and target datasets for this task
        X = feature_dataset[task_name].copy()
        y = target_dataset[task_name].copy()

        # -------------------------------------------------
        # Split into train, validation, and test sets
        # Use stratified split only for classification tasks
        # -------------------------------------------------
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_grouped(
            X, y,
            val_size=0.1,
            test_size=0.2,
            random_state=42,
            stratify=y if task_name == "overloaded_node_classification" else None
        )

        X_test, y_test = remove_augmented_rows_from_test(X_test, y_test, augmented_df, features)
        X_val, y_val = remove_augmented_rows_from_test(X_val, y_val, augmented_df, features)

        # -------------------------------------------------
        # Check for data leakage
        # -------------------------------------------------
        check_node_type_distribution(X_train,X_val,X_test)
        check_data_leakage(X_train,X_val,X_test)

        # -------------------------------------------------
        # Fit MinMaxScaler
        # Skip target scaling for classification tasks
        # -------------------------------------------------
        printmd("## Scalaing")
        if task_name == "overloaded_node_classification":
            scaler_x, scaler_y = fit_minmax_scaler(X_train)
        else:
            scaler_x, scaler_y = fit_minmax_scaler(X_train, y_train,
                                                   regression_cols=task_info["regression_targets"],
                                                  classification_col=task_info["classification_targets"])

        # -------------------------------------------------
        # Save fitted scalers to disk
        # -------------------------------------------------
        save_scalers(scaler_x, scaler_y, task_name, base_path)

        # -------------------------------------------------
        # Apply scaling to train/val/test sets
        # -------------------------------------------------
        (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled
        ) = transform_with_scalers(
            X_train, X_val, X_test,
            scaler_x,
            y_train, y_val, y_test,
            scaler_y,
            regression_cols=task_info["regression_targets"],
            classification_col=task_info["classification_targets"]
        )


        # -------------------------------------------------
        # Optional: Apply any custom transformation on scaled features
        # Skip this step for classification tasks
        # -------------------------------------------------
        if task_name != "overloaded_node_classification":
            X_train_scaled = transform(pd.DataFrame(X_train_scaled, columns=features))
            X_val_scaled = transform(pd.DataFrame(X_val_scaled, columns=features))
            X_test_scaled = transform(pd.DataFrame(X_test_scaled, columns=features))

        # -------------------------------------------------
        # Store scaled datasets and scalers in dictionaries
        # -------------------------------------------------
        x_train_dict[task_name] = X_train_scaled
        x_val_dict[task_name] = X_val_scaled
        x_test_dict[task_name] = X_test_scaled
        x_scalers[task_name] = scaler_x

        y_train_dict[task_name] = y_train_scaled
        y_val_dict[task_name] = y_val_scaled
        y_test_dict[task_name] = y_test_scaled
        y_scalers[task_name] = scaler_y

    # -----------------------------------------------------
    # Return all processed splits and scalers
    # -----------------------------------------------------
    return (
        x_train_dict, x_val_dict, x_test_dict, x_scalers,
        y_train_dict, y_val_dict, y_test_dict, y_scalers
    )


# network creaation and node distribution ------------------------------------------------------------------------------------------------------------------------------------

## Funzioni custom usate da `prepare_data`

# Direttamente:
# - `load_filter_scale`
# - `load_network`
# - `build_nodes_dataframe`
# - `transform`
# - `save_dataset`
# - `aggregate_datasets`

# Indirettamente (tramite `load_filter_scale`):
# - `load_complete_data`
# - `filter_and_remove_outliers`
# - `add_overload_status_ratio`
# - `compute_ram_usage_percentage_theoretical`
# - `extract_features_and_targets`
# - `unify_datasets`
# - `scale_data`

def convert_node_type(node_type: str) -> int:
  node_type_idx = -1
  if node_type == "HEAVY":
    node_type_idx = 0
  elif node_type == "MID":
    node_type_idx = 1
  elif node_type == "LIGHT":
    node_type_idx = 2
  else:
    raise RuntimeError(f"Node type `{node_type}` does not exist")
  return node_type_idx

def assign_node_type(
    towers: pd.DataFrame, node_types: list, rng: np.random.Generator
  ) -> pd.DataFrame:
  nt = [
    convert_node_type(rng.choice(node_types)) for _ in range(len(towers))
  ]
  towers["node_type"] = nt
  return towers

def assign_functions( # attualmente sembra non essere utilizzato i
    towers: pd.DataFrame, 
    functions: list, 
    n_functions_per_node: int,
    rng: np.random.Generator
  ) -> pd.DataFrame:
  functions_per_node = [
    [
      rng.choice(functions) for _ in range(n_functions_per_node)
    ] for _ in range(len(towers))
  ]
  towers["functions"] = functions_per_node
  return towers


def load_network(
    path_to_networks: str, 
    n: int, 
    k: int, 
    #seed: int, 
    simulation: int, 
    rng: np.random.Generator, 
    node_types: list = None,
    functions: list = None,
    n_functions_per_node: int = None
  ):
  towers_file = os.path.join(
    path_to_networks, f"porto_{n}n_{k}k/{simulation}/towers.csv"#path_to_networks, f"porto_{n}n_{k}k/seed{seed}/{simulation}/towers.csv"
  )
  towers = pd.read_csv(towers_file)
  if node_types is not None:
    towers = assign_node_type(towers, node_types, rng)
  if functions is not None:
    towers = assign_functions(towers, functions, n_functions_per_node, rng)
  return towers

# funzione per distribuire i dati ai nodi

def build_nodes_dataframe(
    data_x: pd.DataFrame, data_y: pd.DataFrame, towers: pd.DataFrame
  ) -> dict:
  nodes_dataset = {}
  surely_test_set = {}
  for node_type, node_type_data in data_x.groupby("node_type"):
    # extract the corresponding y values
    node_type_targets = data_y.loc[node_type_data.index]
    # count nodes that have the required type
    nnodes = len(towers[towers["node_type"] == node_type])
    # count available data for that node type (dividing overloaded and 
    # not overloaded)
    nvals = len(node_type_data)
    nvals_overloaded = 0
    if "overloaded_node" in node_type_targets:
      nvals_overloaded = len(
        node_type_targets[node_type_targets["overloaded_node"] == 1]
      )
    nvpn, remainder = [None, None], [None, None]
    nvpn[0], remainder[0] = divmod(nvals - nvals_overloaded, nnodes)
    nvpn[1], remainder[1] = divmod(nvals_overloaded, nnodes)
    # split equally
    for i, node_id in enumerate(
        towers[towers["node_type"] == node_type].index
      ):
      for overload_status in [0, 1]:
        idxs = node_type_targets[
          node_type_targets["overloaded_node"] == overload_status
        ].iloc[
          i * nvpn[overload_status] : (i+1) * nvpn[overload_status]
        ].index
        # add
        if node_id not in nodes_dataset:
          nodes_dataset[node_id] = {
            "x": node_type_data.loc[idxs,:],
            "y": node_type_targets.loc[idxs,:]
          }
        else:
          nodes_dataset[node_id]["x"] = pd.concat(
            [nodes_dataset[node_id]["x"], node_type_data.loc[idxs,:]]
          )
          nodes_dataset[node_id]["y"] = pd.concat(
            [nodes_dataset[node_id]["y"], node_type_targets.loc[idxs,:]]
          )
    # remainder data go into test
    for overload_status in [0, 1]:
      if remainder[overload_status] > 0:
        if node_type not in surely_test_set:
          surely_test_set[node_type] = {
            "x": pd.DataFrame(), "y": pd.DataFrame()
          }
        idxs = node_type_targets[
          node_type_targets["overloaded_node"] == overload_status
        ].iloc[-remainder[overload_status]:].index
        surely_test_set[node_type]["x"] = pd.concat(
          [surely_test_set[node_type]["x"], node_type_data.loc[idxs,:]]
        )
        surely_test_set[node_type]["y"] = pd.concat(
          [surely_test_set[node_type]["y"], node_type_targets.loc[idxs,:]]
        )
  # check coherence
  for node_id, node_data in nodes_dataset.items():
    if (node_data["x"].index != node_data["y"].index).any():
      raise RuntimeError(f"Incoherent dataset for node {node_id}")
  return nodes_dataset, surely_test_set



