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