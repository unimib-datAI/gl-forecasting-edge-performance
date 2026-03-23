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


def add_overload_status_ratio(df, features, target):
  """
  Function to add overload status and overload ratio
  """
  grouped = df.groupby(features)
  df["overloaded_status"] = grouped[target].transform(get_group_status)
  # Ratio of 1’s in the group
  df["overloaded_ratio"] = grouped[target].transform("mean")
  return df


def assign_functions(
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


def assign_node_type(
    towers: pd.DataFrame, node_types: list, rng: np.random.Generator
  ) -> pd.DataFrame:
  nt = [
    convert_node_type(rng.choice(node_types)) for _ in range(len(towers))
  ]
  towers["node_type"] = nt
  return towers


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


def compute_ram_usage_percentage_theoretical(
    df, ram_col="ram_usage_node", node_col="node_type"
  ):
  """Apply ram_usage_to_perc on the DataFrame."""
  # Compute values
  perc_values = df.apply(
      lambda row: ram_usage_to_percentage(row[ram_col], row[node_col]),
      axis=1
  )
  # # Find the index of ram_usage column
  # insert_at = df.columns.get_loc("ram_usage_node_percentage") + 1
  # # Insert new column at that position
  # df.insert(insert_at, "ram_usage_node_perc_theor", perc_values)
  # Replace
  df["ram_usage_node_percentage"] = perc_values
  return df


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


def extract_features_and_targets(
    df: pd.DataFrame, 
    tasks: dict
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Initialize a dictionary to store target datasets
  features_datasets = {}
  target_datasets = {}
  for task_name, task_info in tasks.items():
    x = df[task_info["features"]]
    y = df[task_info["targets"]]
    # Oversampling
    if "classification" in task_name:
      x, y = perform_oversampling(
        x, y, task_info["features"], task_info["targets"]
      )
    features_datasets[task_name] = x
    target_datasets[task_name] = y
  return features_datasets, target_datasets


def fill_NaN(X):
  for col in X:
    if(col.startswith('success_rate_')):
      X.loc[:, col] = X.loc[:, col].fillna(1)
    else:
      X.loc[:, col] = X.loc[:, col].fillna(0)
  return X


def filter_and_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
  # extract only relevant columns
  nf, cf, regression_targets, classification_targets, temp = select_columns(df)
  # loop over regression targets and remove outliers
  df_no_outliers = temp.copy(deep = True)
  grouped = df_no_outliers.groupby(nf + cf)
  threshold = 1
  for target in regression_targets:
    # -- identify outliers
    mean = grouped[target].transform('mean')
    std = grouped[target].transform('std')
    outliers = (
      df_no_outliers[target] > mean + threshold * std
    ) | (
      df_no_outliers[target] < mean - threshold * std
    )
    # -- remove
    df_no_outliers[target] = df_no_outliers[target].where(~outliers, mean)
  # loop over classification targets
  for target in classification_targets:
    new_overloaded = grouped[target].transform(
      lambda x: x.mode().iloc[0]
    )
    df_no_outliers['overloaded_node'] = new_overloaded
  return df_no_outliers, nf, cf, regression_targets, classification_targets


def generate_synthetic_overloaded(df, features, target_col):
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
  print("\n=== Generating synthetic overloaded samples ===")
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
        for val in range(start + 2, 201, 2):
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


def get_group_status(x):
  """
    0 = all 0, 1 = all 1, 2 = mix
  """
  if (x == 0).all():
    return 0
  elif (x == 1).all():
    return 1
  else:
    return 2


def get_node_capacity(node_id: int) -> int:
  # Lookup for node capacities
  GB = 1024 ** 3  # 1 GB in bytes
  capacities = {
      0: 24 * GB,   # Heavy
      1: 16 * GB,   # Mid
      2:  8 * GB    # Light
  }
  return capacities[node_id]


def limit_identical_combinations(X, y, features, target_cols, max_per_combination=1):
  """
  Limit the dataset to at most `max_per_combination` identical feature combinations.

  Parameters
  ----------
  X : pd.DataFrame
      Feature matrix.
  y : pd.DataFrame
      Target(s).
  features : list
      Columns to group by when identifying identical combinations.
  target_cols : list
      Target column names.
  max_per_combination : int
      Maximum allowed identical combinations per group.

  Returns
  -------
  X_limited, y_limited : pd.DataFrame, pd.DataFrame
      Reduced datasets after limiting duplicates.
  """
  print(f"\n=== Limiting to max {max_per_combination} identical combinations ===")
  df_combined = pd.concat([X, y], axis=1)
  df_limited = (
    df_combined.groupby(features, group_keys=False)
    .apply(lambda x: x.head(max_per_combination))
    .reset_index(drop=True)
  )
  X_limited = df_limited[features]
  y_limited = df_limited[target_cols]
  print(f"Original row count: {len(df_combined)}")
  print(f"After limiting: {len(df_limited)} (removed {len(df_combined) - len(df_limited)})")
  print("\nClass distribution after limiting:")
  print(y_limited.value_counts())
  return X_limited, y_limited


def load_complete_data(
    node_types: list, 
    path_to_csvs: str,
    keep_baselines: bool = False
  ) -> pd.DataFrame:
  df = pd.DataFrame()
  for node_type in node_types:
    # Retrieve all files in the output folder
    node_data_path = os.path.join(path_to_csvs, node_type)
    file_csv = [
      file for file in os.listdir(node_data_path) if file.endswith('.csv')
    ]
    # Create the dataframe by concatenating all read files
    dataframes = []
    for file in file_csv:
      file_path = os.path.join(node_data_path, file)
      df_temp = pd.read_csv(file_path)
      # Remove the columns in the dataframe that begin with "function_"
      df_temp.drop(
        columns=[col for col in df_temp if col.startswith('function_')], 
        inplace=True
      )
      # Add the column "node_type" and assign the value of 'type' to all rows
      df_temp["node_type"] = convert_node_type(node_type)
      dataframes.append(df_temp)
    # Concat
    df = pd.concat([df, *dataframes], axis=0, ignore_index=True)
  # Remove NaN
  df = fill_NaN(df)
  # Extract list of functions
  rates_col = [col for col in df if col.startswith('rate')]
  # Remove baselines
  if not keep_baselines:
    df = remove_baselines(df, rates_col)
  return df


def perform_oversampling(x, y, features, target_cols, limit_combinations = False):
  """
  Custom domain-aware oversampling replacing SMOTENC.

  Steps:
  1. (optional) Limit identical feature combinations.
  2. Generate synthetic overloaded samples per node_type.
  3. Combine and return balanced dataset.

  Parameters
  ----------
  X : pd.DataFrame
      Feature matrix.
  y : pd.DataFrame
      Target(s).
  features : list
      Feature columns used for grouping and synthesis.
  target_cols : list
      Target column(s).

  Returns
  -------
  X_resampled, y_resampled : pd.DataFrame, pd.DataFrame
      Balanced feature and target datasets.
  """
  df = pd.concat([x, y], axis = 1)
  if limit_combinations:
    # Step 1: Apply limiting rule
    X_limited, y_limited = limit_identical_combinations(
      x, y, features, target_cols
    )
    # Step 2: Combine for rebalancing
    df = pd.concat([X_limited, y_limited], axis=1)
  target_col = target_cols[0]
  # Step 3: Generate new samples
  df_augmented = generate_synthetic_overloaded(df, features, target_col)
  # # Optionally save the augmented rows
  # os.makedirs(PATH_TO_AUGMENTED_ROWS, exist_ok=True)
  # df_augmented.to_csv(PATH_TO_AUGMENTED_ROWS + "augmented_overloaded_samples.csv", index=False)
  # print("Augmented rows saved to 'augmented_overloaded_samples.csv'")
  # Step 4: Final balanced dataset
  df_balanced = pd.concat([df, df_augmented], ignore_index=True)
  X_res = df_balanced[features]
  y_res = df_balanced[[target_col]]
  print("\nFinal class distribution after rebalancing:")
  print(y_res.value_counts())
  print("\nFinal class distribution after rebalancing (node_type-wise):")
  print(
    df_balanced.groupby("node_type")["overloaded_node"].value_counts().unstack(
      fill_value = 0
    )
  )
  print("\nFinal node_type distribution:")
  print(X_res["node_type"].value_counts())
  return X_res, y_res


def ram_usage_to_percentage(ram_usage, node_type):
  """Return RAM utilization % given usage (bytes) and node type."""
  return (ram_usage / get_node_capacity(node_type)) * 100


# Remove rows where all specified columns have value 0.
def remove_baselines(df, features):
  df = df.copy()
  return df[~(df[features] == 0).all(axis=1)]


def scale_data(
    feature_datasets: dict, 
    target_datasets: dict,
    tasks: dict,
    classification_targets: list,
    scalers_output_path: str
  ) -> Tuple[dict, dict, dict]:
  y_scalers = {}
  x_scaled_dict = {}
  y_scaled_dict = {}
  # loop over tasks
  for task_name in tasks:
    # extract x and y
    x = feature_datasets[task_name].copy().drop("node_type", axis = "columns")
    node_type = feature_datasets[task_name]["node_type"]
    regression_y = target_datasets[task_name].copy().drop(
      classification_targets, axis = 1
    )
    classification_y = target_datasets[
      task_name
    ].copy()[classification_targets]
    # scale x (except the node type!)
    scaler_x = MinMaxScaler()
    scaler_x.fit(x)
    x_scaled = pd.DataFrame(scaler_x.transform(x), columns = x.columns)
    x_scaled["node_type"] = node_type
    # scale y (for regression only!)
    scaler_y = MinMaxScaler(feature_range = (1,2))
    scaler_y.fit(regression_y)
    y_scalers[task_name] = scaler_y
    y_scaled = pd.DataFrame(
      scaler_y.transform(regression_y), columns = regression_y.columns
    )
    y_scaled = pd.concat([y_scaled, classification_y], axis = 1)
    # Save the scaler for x
    scaler_x_path = os.path.join(scalers_output_path, 'scaler_x')
    os.makedirs(scaler_x_path, exist_ok = True)
    joblib.dump(
      scaler_x, os.path.join(scaler_x_path, f"{task_name}_features.joblib")
    )
    # Save the scaler for y
    if not "classification" in task_name:
      scaler_y_path = os.path.join(scalers_output_path, 'scaler_y')
      os.makedirs(scaler_y_path, exist_ok = True)
      joblib.dump(
        scaler_y, os.path.join(scaler_y_path, f"{task_name}.joblib")
      )
    # save
    x_scaled_dict[task_name] = x_scaled
    y_scaled_dict[task_name] = y_scaled
  return x_scaled_dict, y_scaled_dict, y_scalers


def select_columns(df) -> Tuple[list, list, list, list, pd.DataFrame]:
  # regression_targets = [
  #   col for col in df if (
  #     col.startswith('cpu_usage_') or
  #       col.startswith('ram_usage_') or
  #         col.startswith('medium_latency')
  #   ) and 'idle' not in col
  # ]# or col.startswith('replica')
  regression_targets = [
    col for col in df if (
      col.startswith('cpu_usage_node') or
        col.startswith('ram_usage_node')
    ) and 'idle' not in col
  ]# or col.startswith('replica')
  classification_targets = [
    col for col in df if (
      col.startswith('overloaded_node')
    ) and 'idle' not in col
  ]
  numerical_features = [
    col for col in df if col.startswith('rate_')
  ]
  numerical_features.sort()
  categorical_features = ["node_type"]
  return (
    numerical_features, 
    categorical_features, 
    regression_targets, 
    classification_targets,
    df[
      numerical_features + 
        categorical_features + 
          regression_targets + 
            classification_targets
    ]
  )


def transform(df, rate_columns: list, pad_value: float = 0.0):
  """
  Transform the input dataframe into padded, variable-length sequences.

  Each sample becomes a sequence of steps where each 
  step = [rate, method_onehot(6), node_type].
  Steps with zero rate are removed. Then sequences are padded to the same 
  length.

  Returns:
      X_padded: np.ndarray, shape (num_samples, max_seq_len, 8)
      mask: np.ndarray, shape (num_samples, max_seq_len), 1 for valid, 0 for 
      padded
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
  # Build sequences
  X_sequences = []
  for i in range(num_samples):
    node_val = node_type[i]
    rate_values = df.loc[i, rate_columns].values  # shape (6,)
    # Keep only non-zero steps
    nonzero_mask = rate_values != 0
    if np.any(nonzero_mask):
      rate_seq = rate_values[nonzero_mask].reshape(-1, 1)
      method_seq = method_onehot[nonzero_mask]
      node_seq = np.full((np.sum(nonzero_mask), 1), node_val)
      seq = np.concatenate([rate_seq, method_seq, node_seq], axis=1)  # (L_i, 8)
      X_sequences.append(seq)
  # Pad all sequences to same length (post-padding with zeros)
  max_len = max(len(seq) for seq in X_sequences)
  X_padded = pad_sequences(
    X_sequences, 
    maxlen=max_len, 
    dtype='float32', 
    padding='post', 
    value=pad_value
  )
  # Create mask (1 for real steps, 0 for padded)
  mask = np.array(
    [[1]*len(seq) + [0]*(max_len - len(seq)) for seq in X_sequences], 
    dtype='float32'
  )
  # Drop node_type column if you want to reuse df
  df = df.drop(columns=['node_type'])
  return X_padded


def load_filter_scale(
    node_types: list, path_to_csvs: str, plot_info: bool = False
  ):
  output_csvs_path = "../data/preprocessed"
  base_filename = "node_only_multi_task"
  multi_target_x, multi_target_y, tasks, unified_tasks = [None] * 4
  if (
      os.path.exists(output_csvs_path) and 
        os.path.exists(
          os.path.join(output_csvs_path, f"{base_filename}_x.csv")
        ) and
          os.path.exists(
            os.path.join(output_csvs_path, f"{base_filename}_y.csv")
          ) and
            os.path.exists(
              os.path.join(output_csvs_path, f"{base_filename}_tasks.json")
            ) and
              os.path.exists(
                os.path.join(output_csvs_path, f"{base_filename}_utasks.json")
              )
    ):
    multi_target_x = pd.read_csv(
      os.path.join(output_csvs_path, f"{base_filename}_x.csv")
    )
    multi_target_y = pd.read_csv(
      os.path.join(output_csvs_path, f"{base_filename}_y.csv")
    )
    with open(
        os.path.join(output_csvs_path, f"{base_filename}_tasks.json"), "r"
      ) as istream:
      tasks = json.load(istream)
    with open(
        os.path.join(output_csvs_path, f"{base_filename}_utasks.json"), "r"
      ) as istream:
      unified_tasks = json.load(istream)
  else:
    os.makedirs(output_csvs_path, exist_ok = True)
    # load data
    df = load_complete_data(node_types, path_to_csvs)
    # remove outliers and keep only relevant columns
    (
      df_no_outliers, 
      numerical_features, 
      categorical_features, 
      regression_targets, 
      classification_targets
    ) = filter_and_remove_outliers(df)
    # add more details on overloaded state
    df_no_outliers = add_overload_status_ratio(
      df_no_outliers, 
      numerical_features + categorical_features, 
      "overloaded_node"
    )
    # compute percentage ram usage
    df_no_outliers = compute_ram_usage_percentage_theoretical(df_no_outliers)
    # plot distributions (if required)
    if plot_info:
      # -- plot node type distribution
      plot_node_type_distribution(df_no_outliers)
      # -- plot overloaded node distribution
      plot_overloaded_node_distribution(df_no_outliers)
      plot_overloaded_node_distribution_by_type(df_no_outliers)
    # feature-target partitioning and class balance via oversampling
    tasks = {
      "Multi_Task_regression": {
        "features": numerical_features + categorical_features,
        "targets": regression_targets
      },
      "Multi_Task_classification": {
        "features": numerical_features + categorical_features,
        "targets": classification_targets
      }
    }
    feature_datasets, target_datasets = extract_features_and_targets(
      df_no_outliers, tasks
    )
    # create unified dataset
    unified_df, unified_tasks = unify_datasets(
      feature_datasets, 
      target_datasets, 
      tasks, 
      numerical_features + categorical_features
    )
    (
      unified_feature_datasets, 
      unified_target_datasets
    ) = extract_features_and_targets(
      unified_df, 
      unified_tasks
    )
    # scale
    X_scaled, y_scaled, _ = scale_data(
      unified_feature_datasets,
      unified_target_datasets,
      unified_tasks,
      classification_targets,
      "../output/scalers/functions_multi_target"
    )
    multi_target_x = X_scaled["Multi_Task"]
    multi_target_y = y_scaled["Multi_Task"]
    # save
    multi_target_x.to_csv(
      os.path.join(output_csvs_path, f"{base_filename}_x.csv"), index = False
    )
    multi_target_y.to_csv(
      os.path.join(output_csvs_path, f"{base_filename}_y.csv"), index = False
    )
    with open(
        os.path.join(output_csvs_path, f"{base_filename}_tasks.json"), "w"
      ) as ost:
      ost.write(json.dumps(tasks, indent = 2))
    with open(
        os.path.join(output_csvs_path, f"{base_filename}_utasks.json"), "w"
      ) as ost:
      ost.write(json.dumps(unified_tasks, indent = 2))
  return multi_target_x, multi_target_y, tasks, unified_tasks


def load_network(
    path_to_networks: str, 
    n: int, 
    k: int, 
    seed: int, 
    simulation: int, 
    rng: np.random.Generator, 
    node_types: list = None,
    functions: list = None,
    n_functions_per_node: int = None
  ):
  towers_file = os.path.join(
    path_to_networks, f"porto_{n}n_{k}k/seed{seed}/{simulation}/towers.csv"
  )
  towers = pd.read_csv(towers_file)
  if node_types is not None:
    towers = assign_node_type(towers, node_types, rng)
  if functions is not None:
    towers = assign_functions(towers, functions, n_functions_per_node, rng)
  return towers


def plot_node_type_distribution(df):
  """
  Displays the number of data for each node type in a DataFrame using a 
  bar chart.
  :param df: DataFrame containing the 'node_type' column with the node types.
  """
  # Count of occurrences for each node type
  node_counts = df['node_type'].value_counts().sort_index()
  # Bar chart
  fig, axes = plt.subplots(1, 1, figsize=(10, 6))
  node_counts.plot(kind='bar',ax=axes)
  plt.title('Distribution of Number of Data by Node Type')
  plt.xlabel('Node Type (0 = Heavy, 1 = Mid, 2 = Light)')
  plt.ylabel('Number of Rows')
  plt.xticks(rotation=0)  # Maintains names of horizontal node types
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  for i in axes.containers:
    axes.bar_label(i)
  plt.show()


def plot_overloaded_node_distribution(df):
  """
  Displays the number of rows where 'overloaded_node' is 1 and the number 
  where it is 0.
  :param df: DataFrame containing column 'overloaded_node'.
  """
  # Count of occurrences of 0 and 1 in the 'overloaded_node' column
  overloaded_counts = df['overloaded_node'].value_counts()
  # Bar chart
  fig, axes = plt.subplots(1, 1, figsize=(8, 5))
  overloaded_counts.plot(kind='bar',ax=axes)
  plt.title('Distribution of Overloaded Nodes')
  plt.xlabel('Overload Status (0 = No, 1 = Yes)')
  plt.ylabel('Number of Lines')
  plt.xticks(rotation=0)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  for i in axes.containers:
    axes.bar_label(i)
  plt.show()


def plot_overloaded_node_distribution_by_type(df):
  """
  Displays a grouped bar chart showing the count of overloaded (1) and 
  non-overloaded (0) nodes for each node_type category, with value labels on 
  top of bars.
  """
  # Group by node_type and overloaded_node, then count
  counts = df.groupby(['node_type', 'overloaded_node']).size().reset_index(
    name='count'
  )
  # Convert overloaded_node to readable labels
  counts['overloaded_node'] = counts['overloaded_node'].map(
    {0: 'Not Overloaded', 1: 'Overloaded'}
  )
  # Define custom colors
  custom_palette = {
    'Not Overloaded': '#29af7f',   # viridis green
    'Overloaded': '#3b528b'       # viridis blue
  }
  # Plot
  plt.figure(figsize=(10, 6))
  ax = sns.barplot(
    data=counts,
    x='node_type',
    y='count',
    hue='overloaded_node',
    palette=custom_palette
  )
  # Add bar labels (numbers on top)
  for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3, fontsize=10)
  plt.title('Distribution of Overloaded Nodes by Node Type')
  plt.xlabel('Node Type')
  plt.ylabel('Number of Nodes')
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.legend(title='Overload Status')
  plt.tight_layout()
  plt.show()


def split_test_nodewise(X_test_dict, Y_test_dict, tasks):
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
    node_types = [0.0, 0.5, 1.0]
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


def unify_datasets(
    features_dict, targets_dict, tasks,feature_columns,groupby_col='node_type'
  ):
  """
  Create complete datasets for regression and classification tasks,
  then perform an outer join on the given feature columns.

  Parameters
  ----------
  features_dict : dict
      Dictionary containing features for each task.
      Example:
      {
          'Multi_Target_regression': X_reg,
          'overloaded_node_classification': X_cls
      }

  targets_dict : dict
      Dictionary containing targets for each task.
      Example:
      {
          'Multi_Target_regression': y_reg,
          'overloaded_node_classification': y_cls
      }

  feature_columns : list
      List of column names to use as join keys for merging the two datasets.

  Returns
  -------
  final_df : pd.DataFrame
      Outer-joined DataFrame combining regression and classification datasets.
  """
  # Step 1: Combine features and targets per task
  complete_datasets = {}
  complete_tasks = {"Multi_Task": {
    "features": feature_columns,
    "targets": []
  }}
  for task in tasks:
    if task not in features_dict or task not in targets_dict:
        raise KeyError(f"Missing data for task: {task}")
    x = features_dict[task]
    y = targets_dict[task]
    # Ensure DataFrame
    X_df = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x.copy()
    # Ensure target DataFrame
    if isinstance(y, pd.DataFrame):
      y_df = y.copy()
    elif isinstance(y, pd.Series):
      y_df = y.to_frame("target")
    else:
      y_df = pd.DataFrame(
        y, columns = [
          f"target_{i}" for i in range(y.shape[1])
        ] if len(y.shape) > 1 else ["target"]
      )
    # Merge horizontally
    complete_datasets[task] = pd.concat(
      [X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1
    )
    # Record task
    complete_tasks["Multi_Task"]["targets"] += tasks[task]["targets"]
  # Step 2: Perform outer join on feature columns
  df_reg = complete_datasets["Multi_Task_regression"]
  df_cls = complete_datasets["Multi_Task_classification"]
  # Check if all feature columns exist in both
  missing_in_reg = [col for col in feature_columns if col not in df_reg.columns]
  missing_in_cls = [col for col in feature_columns if col not in df_cls.columns]
  if missing_in_reg or missing_in_cls:
    raise ValueError(
      f"Missing join columns: "
      f"regression missing {missing_in_reg}, "
      f"classification missing {missing_in_cls}"
    )
  # Outer join
  merged_df = pd.merge(
    df_reg, 
    df_cls, 
    on=feature_columns, 
    how='outer', 
    suffixes=('_reg', '_cls')
  )
  final_df = merged_df.copy()
  # Step 3: Fill NaN values by node_type group max
  if groupby_col not in final_df.columns:
      raise KeyError(
        f"groupby_col '{groupby_col}' not found in merged DataFrame."
      )
  # Identify numeric columns to fill
  numeric_cols = final_df.select_dtypes(include=[np.number]).columns
  # Fill NaN values by groupwise max
  final_df[numeric_cols] = final_df.groupby(
    groupby_col
  )[numeric_cols].transform(lambda g: g.fillna(g.max()))
  validate_groupwise_max_fill(
    merged_df, 
    final_df, 
    groupby_col='node_type', 
    numeric_cols=None, 
    verbose=True
  )

  return final_df, complete_tasks


def validate_groupwise_max_fill(
    original_df, 
    filled_df, 
    groupby_col='node_type', 
    numeric_cols=None, 
    verbose=True
  ):
  """
  Validate that NaN values were filled correctly with groupwise column-wise max,
  and confirm that no NaNs remain in the filled DataFrame.

  Parameters
  ----------
  original_df : pd.DataFrame
      DataFrame BEFORE filling NaNs.
  filled_df : pd.DataFrame
      DataFrame AFTER filling NaNs.
  groupby_col : str, default='node_type'
      Column used for grouping during fill.
  numeric_cols : list or None
      List of numeric columns to check. If None, detected automatically.
  verbose : bool, default=True
      Print detailed results.

  Returns
  -------
  validation_summary : dict
      {
          'validated_columns': [...],
          'correct_fills': int,
          'incorrect_fills': list of dicts,
          'remaining_nans': dict(column_name -> count)
      }
  """

  # Auto-detect numeric columns if not provided
  if numeric_cols is None:
    numeric_cols = filled_df.select_dtypes(
      include=[np.number]
    ).columns.tolist()
  validation_summary = {
    "validated_columns": numeric_cols,
    "correct_fills": 0,
    "incorrect_fills": [],
    "remaining_nans": {}
  }
  # Step 1: Compute groupwise maxima from original data
  groupwise_max = original_df.groupby(
    groupby_col
  )[numeric_cols].transform('max')
  # Step 2: Verify that each filled NaN matches the expected groupwise max
  for col in numeric_cols:
    for idx in original_df.index:
      group_val = original_df.at[idx, groupby_col]
      orig_val = original_df.at[idx, col]
      filled_val = filled_df.at[idx, col]
      # Only validate if this was a filled NaN originally
      if pd.isna(orig_val) and not pd.isna(groupwise_max.at[idx, col]):
        expected_val = groupwise_max.at[idx, col]
        if np.isclose(filled_val, expected_val, equal_nan=True):
          validation_summary["correct_fills"] += 1
        else:
          validation_summary["incorrect_fills"].append({
            "index": idx,
            "group": group_val,
            "column": col,
            "filled_value": filled_val,
            "expected_value": expected_val
          })
  # Step 3: Check for any remaining NaN values in the filled DataFrame
  nan_counts = filled_df[numeric_cols].isna().sum()
  remaining_nans = nan_counts[nan_counts > 0].to_dict()
  validation_summary["remaining_nans"] = remaining_nans
  # Step 4: Reporting
  if verbose:
    total_checked = validation_summary["correct_fills"] + len(
      validation_summary["incorrect_fills"]
    )
    print("\nValidation Summary:")
    print(
      f"Checked {total_checked} filled entries "
      f"across {len(numeric_cols)} numeric columns."
    )
    print(f"Correct fills: {validation_summary['correct_fills']}")
    print(f"Incorrect fills: {len(validation_summary['incorrect_fills'])}")
    if validation_summary["incorrect_fills"]:
      print("Sample incorrect fills:")
      for err in validation_summary["incorrect_fills"][:5]:  # show first few
        print(f"   - {err}")
    if remaining_nans:
      print("\nRemaining NaNs detected after fill:")
      for col, count in remaining_nans.items():
        print(f"   {col}: {count} NaNs")
    else:
      print("\nNo NaN values remain in the filled DataFrame.")
  return validation_summary



def prepare_data(
    node_types: list, 
    path_to_csvs: str, 
    path_to_networks: str, 
    base_output_folder: str,
    n: int, 
    k: int,
    seed: int,
    simulation: int,
    test_perc: float,
    val_perc_on_train: float
  ):
  # prepare output folder
  output_folder = os.path.join(
    base_output_folder, f"{n}n-{k}k", f"seed{seed}", str(simulation)
  )
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    # load, filter and scale original data
    print("load, filter and scale original data")
    multi_target_x, multi_target_y, tasks, unified_tasks = load_filter_scale(
      node_types, path_to_csvs
    )
    print("...done")
    # load nodes network
    print("load nodes network")
    rng = np.random.default_rng(seed)
    network = load_network(
      path_to_networks, 
      n, 
      k, 
      seed, 
      simulation, 
      rng,
      node_types = node_types
      # functions = [col for col in multi_target_x if col.startswith('rate')],
      # n_functions_per_node = 3
    )
    print("...done")
    # build nodes dataframes
    print("build nodes dataframes")
    nodes_dataframe, test_set = build_nodes_dataframe(
      multi_target_x, multi_target_y, network
    )
    print("...done")
    # split and save
    print("split and save")
    prepared_nodes_dataset = {}
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for node, data in nodes_dataframe.items():
      group_indices = data["x"].groupby(list(data["x"].columns)).groups
      group_keys = pd.DataFrame(group_indices.keys())
      # -- test
      test_idxs = group_keys.sample(frac = test_perc, random_state = seed)
      test_data = {"x": pd.DataFrame(), "y": pd.DataFrame()}
      for idx in test_idxs.index:
        key = tuple(test_idxs.loc[idx,:])
        test_data["x"] = pd.concat(
          [test_data["x"], data["x"].loc[group_indices[key], :]]
        )
        test_data["y"] = pd.concat(
          [test_data["y"], data["y"].loc[group_indices[key], :]]
        )
      train_val_idxs = group_keys.drop(test_idxs.index)
      # -- validation
      val_idxs = train_val_idxs.sample(
        frac = val_perc_on_train, random_state = seed
      )
      val_data = {"x": pd.DataFrame(), "y": pd.DataFrame()}
      for idx in val_idxs.index:
        key = tuple(val_idxs.loc[idx,:])
        val_data["x"] = pd.concat(
          [val_data["x"], data["x"].loc[group_indices[key], :]]
        )
        val_data["y"] = pd.concat(
          [val_data["y"], data["y"].loc[group_indices[key], :]]
        )
      # -- train
      train_idxs = train_val_idxs.drop(val_idxs.index)
      train_data = {"x": pd.DataFrame(), "y": pd.DataFrame()}
      for idx in train_idxs.index:
        key = tuple(train_idxs.loc[idx,:])
        train_data["x"] = pd.concat(
          [train_data["x"], data["x"].loc[group_indices[key], :]]
        )
        train_data["y"] = pd.concat(
          [train_data["y"], data["y"].loc[group_indices[key], :]]
        )
      train_data["x"].reset_index(inplace = True, drop = True)
      train_data["y"].reset_index(inplace = True, drop = True)
      val_data["x"].reset_index(inplace = True, drop = True)
      val_data["y"].reset_index(inplace = True, drop = True)
      test_data["x"].reset_index(inplace = True, drop = True)
      test_data["y"].reset_index(inplace = True, drop = True)
      # transform
      features = [c for c in train_data["x"].columns if c != "node_type"]
      X_train = transform(train_data["x"], features)
      X_val = transform(val_data["x"], features)
      X_test = transform(test_data["x"], features)
      Y_train = train_data["y"].to_numpy()
      Y_val = val_data["y"].to_numpy()
      Y_test = test_data["y"].to_numpy()
      # save
      prepared_nodes_dataset[node] = (
        X_train, Y_train, X_val, Y_val, X_test, Y_test
      )
      # -- to file
      save_dataset(
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        output_folder,
        node
      )
      # prepare datasets to train centralized model
      train_datasets.append((X_train, Y_train))
      val_datasets.append((X_val, Y_val))
      test_datasets.append((X_test, Y_test))
    # aggregate and save centralized dataset
    centralized_train_data = aggregate_datasets(train_datasets)
    centralized_val_data = aggregate_datasets(val_datasets)
    centralized_test_data = aggregate_datasets(test_datasets)
    save_dataset(
      *centralized_train_data,
      *centralized_val_data,
      *centralized_test_data,
      output_folder,
      "centralized"
    )
    # -- save also tasks names
    with open(os.path.join(output_folder, "tasks.json"), "w") as ost:
      ost.write(json.dumps(tasks, indent = 2))
    with open(os.path.join(output_folder, "unified_tasks.json"), "w") as ost:
      ost.write(json.dumps(unified_tasks, indent = 2))
    print("...done")
  else:
    print(f"WARNING: Output folder {output_folder} already exists!")
  return output_folder


if __name__ == "__main__":
  path_to_csvs = "../data/raw/source_domain/"
  path_to_networks = "../data/networks"
  base_output_folder = "../experiments"
  node_types = ["LIGHT", "MID", "HEAVY"]
  n = 10
  k = 3
  seed = 4850
  simulations = range(10)
  for simulation in simulations:
    print(f"{20*'-'} {simulation} {20*'-'}")
    prepare_data(
      node_types, 
      path_to_csvs, 
      path_to_networks, 
      base_output_folder, 
      n, 
      k, 
      seed, 
      simulation, 
      0.1, 
      0.1
    )
