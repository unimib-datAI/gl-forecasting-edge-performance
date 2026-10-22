#!/usr/bin/env python3
"""
Compare independently trained node models on every node's training data.

Expected layout:
    BASE_DIR/
      0_data.json ... N_data.json
      results_SEED/
        0_single.keras ... N_single.keras
        0_single_pred.json ... N_single_pred.json

Each data JSON must contain X_train and Y_train. X has shape
(samples, sequence_length, features), with node_type in the last feature.
Y has three regression targets followed by the overload label.

For every model i and dataset j:
  * replace node_type with -1 on valid input steps in X_train[j];
  * prepare Janossy input with six sequence permutations;
  * predict with model i;
  * compare predictions to Y_train[j], ignoring non-finite regression labels;
  * save pairwise distances and exp(-aggregate_distance) similarity.

Also runs a diagnostic for each node's own training data: predict with 
node_type=-1 and compare the regression predictions with the saved 
*_single_pred.json values.
The diagnostic assumes the saved JSON's "train" object maps sample indices
(e.g. "0", "1", ...) to three regression predictions.
"""
from __future__ import annotations

from itertools import permutations
from typing import Any
import argparse
import random
import json
import os

from keras.api.saving import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--base-dir",
    type = str,
    required = True,
    help = "Folder containing N_data.json and results folder"
  )
  parser.add_argument(
    "--results-seed",
    type = int,
    required = True,
    help = "Seed used to compute results (assumed to be saved in results_SEED)"
  )
  parser.add_argument(
    "--nodes", 
    nargs = "+", 
    default = "all"
  )
  parser.add_argument(
    "--seed", type=int, default=1234
  )
  parser.add_argument(
    "--regression-weight",
    type = float,
    default = 0.5,
    help = "Weight for mean regression MSE",
  )
  return parser.parse_args()


def load_node_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
  with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  if "X_train" not in data or "Y_train" not in data:
    raise KeyError(f"{data_path} must contain X_train and Y_train.")
  X = np.asarray(data["X_train"], dtype=np.float32)
  Y = np.asarray(data["Y_train"], dtype=np.float64)
  if X.ndim != 3:
    raise ValueError(
      f"{data_path}: expected X_train to be 3-D; got {X.shape}."
    )
  if Y.ndim != 2 or len(X) != len(Y):
    raise ValueError(
      f"{data_path}: X_train/Y_train have "
      f"incompatible shapes {X.shape}/{Y.shape}."
    )
  if X.shape[2] < 2 or Y.shape[1] < 2:
    raise ValueError(f"{data_path}: unexpected feature/target dimensions.")
  return X, Y


def replace_node_type_with_minus_one(X: np.ndarray) -> np.ndarray:
  """Set the last feature to -1 only on real (non-padding) sequence steps."""
  result = np.array(X, copy=True)
  # Same validity rule as Node._extract_node_type: at least one feature 
  # before node_type is nonzero. Padding rows retain their original last 
  # feature.
  valid_steps = np.any(result[:, :, :-1] != 0, axis=-1)
  result[:, :, -1][valid_steps] = -1.0
  return result


def prepare_janossy_input(
    X: np.ndarray, num_permutations: int, rng_seed: int
  ) -> np.ndarray:
  """
  Reproduce utils.janossy.prepare_janossy_test_input's output shape:
  (samples, permutations, sequence_length, features).
  A fixed seed makes this analysis reproducible.
  """
  random_state = random.getstate()
  numpy_state = np.random.get_state()
  random.seed(rng_seed)
  np.random.seed(rng_seed)
  try:
    n_samples, seq_len, feat_dim = X.shape
    all_perms = list(permutations(range(seq_len))) if seq_len <= 6 else None
    prepared = np.zeros(
      (n_samples, num_permutations, seq_len, feat_dim), dtype=np.float32
    )
    for i in range(n_samples):
      if all_perms and len(all_perms) >= num_permutations:
        selected = random.sample(all_perms, num_permutations)
      else:
        selected = [np.random.permutation(seq_len) for _ in range(num_permutations)]
      for j, perm in enumerate(selected):
        prepared[i, j] = X[i, list(perm)]
    return prepared
  finally:
    random.setstate(random_state)
    np.random.set_state(numpy_state)


def predict(model: Any, X: np.ndarray, num_permutations: int, seed: int):
  X_prepared = prepare_janossy_input(X, num_permutations, seed)
  outputs = model.predict(X_prepared, verbose=0)
  if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
    raise ValueError(
      "Expected a two-head model returning [regression_predictions, "
      "classification_probabilities]."
    )
  reg_pred = np.asarray(outputs[0], dtype=np.float64)
  cls_pred = np.asarray(outputs[1], dtype=np.float64).reshape(-1)
  if reg_pred.ndim != 2:
    raise ValueError(f"Expected 2-D regression predictions; got {reg_pred.shape}.")
  if len(reg_pred) != len(X) or len(cls_pred) != len(X):
    raise ValueError(
      f"Prediction/sample count mismatch: X={len(X)}, reg={reg_pred.shape}, "
      f"cls={cls_pred.shape}."
    )
  return reg_pred, cls_pred


def calculate_distances(
    Y_true: np.ndarray,
    reg_pred: np.ndarray,
    cls_prob: np.ndarray,
    regression_weight: float,
  ) -> dict[str, float | int]:
  """MSE distances; regression NaNs are omitted target-by-target."""
  if Y_true.shape[1] != reg_pred.shape[1] + 1:
    raise ValueError(
      f"Y has {Y_true.shape[1]} columns but regression output has "
      f"{reg_pred.shape[1]} targets; expected one extra classification column."
    )
  result: dict[str, float | int] = {}
  per_target_mses = []
  valid_reg_entries = 0
  for k in range(reg_pred.shape[1]):
    truth_k = Y_true[:, k]
    pred_k = reg_pred[:, k]
    valid = np.isfinite(truth_k) & np.isfinite(pred_k)
    valid_reg_entries += int(valid.sum())
    mse = float(
      np.mean((truth_k[valid] - pred_k[valid]) ** 2)
    ) if valid.any() else np.nan
    result[f"regression_mse_target_{k}"] = mse
    if np.isfinite(mse):
      per_target_mses.append(mse)
  avg_reg_mse = float(np.mean(per_target_mses)) if per_target_mses else np.nan
  cls_true = Y_true[:, -1]
  valid_cls = np.isfinite(cls_true) & np.isfinite(cls_prob)
  cls_mse = (
    float(np.mean((cls_true[valid_cls] - cls_prob[valid_cls]) ** 2))
    if valid_cls.any()
    else np.nan
  )
  # Combine available components only. If one task has no usable labels, use
  # the other task's distance rather than making the entire score undefined.
  components = []
  weights = []
  if np.isfinite(avg_reg_mse):
    components.append(avg_reg_mse)
    weights.append(regression_weight)
  if np.isfinite(cls_mse):
    components.append(cls_mse)
    weights.append(1.0 - regression_weight)
  if not components:
    aggregate = np.nan
  else:
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.sum() == 0:
      aggregate = float(np.mean(components))
    else:
      aggregate = float(np.average(components, weights=weights_arr))
  result.update(
    avg_regression_mse = avg_reg_mse,
    classification_mse = cls_mse,
    aggregate_distance = aggregate,
    similarity_score = float(
      np.exp(-aggregate)
    ) if np.isfinite(aggregate) else np.nan,
    n_valid_regression_entries = valid_reg_entries,
    n_valid_classification_samples = int(valid_cls.sum()),
  )
  return result


def load_saved_predictions(
    path: str, node_id: int, target: str = "train"
  ) -> dict[int, np.ndarray]:
  """
  Read saved predictions
  0: regression predictions
  1: classification predictions
  """
  with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
  if target not in obj:
    raise KeyError(f"{path} has no '{target}' section.")
  target_obj = obj[target]
  regression_predictions = np.asarray(target_obj["0"])
  classification_predictions_raw = np.asarray(target_obj["1"])
  classification_predictions = np.array(
    [
      0 if float(np.asarray(p).squeeze()) < 0.5 else 1
      for p in classification_predictions_raw
    ],
    dtype = int,
  ).reshape(-1, 1)
  return (
    regression_predictions, 
    classification_predictions, 
    classification_predictions_raw
  )


def run_diagnostic(
    node_id: int,
    X: np.ndarray,
    model: Any,
    saved_predictions_path: Path,
    num_permutations: int,
    seed: int,
  ) -> dict[str, Any]:
  X_minus_one = replace_node_type_with_minus_one(X)
  reg_pred, cls_prob = predict(model, X_minus_one, num_permutations, seed)
  saved_reg, _, saved_cls_raw = load_saved_predictions(
    saved_predictions_path, node_id, "train"
  )
  # Compare
  rows: dict[str, Any] = {
    "node": node_id,
    "n_current_train_samples": len(reg_pred),
    "n_saved_prediction_samples": len(saved_reg),
    "saved_predictions_file": str(saved_predictions_path),
  }
  diff_reg = np.asarray(reg_pred) - saved_reg
  diff_cls = np.asarray(cls_prob) - saved_cls_raw
  rows["n_compared_samples"] = reg_pred.shape[0]
  rows["regression_mse_vs_saved"] = float(np.mean(diff_reg ** 2))
  rows["regression_mae_vs_saved"] = float(np.mean(np.abs(diff_reg)))
  rows["regression_rmse_vs_saved"] = float(np.sqrt(np.mean(diff_reg ** 2)))
  rows["cls_prob_mse_vs_saved"] = float(np.mean(diff_cls ** 2))
  rows["cls_prob_mae_vs_saved"] = float(np.mean(np.abs(diff_cls)))
  rows["cls_prob_rmse_vs_saved"] = float(np.sqrt(np.mean(diff_cls ** 2)))
  for k in range(diff_reg.shape[1]):
    rows[f"mse_vs_saved_reg_target_{k}"] = float(
      np.mean(diff_reg[:, k] ** 2)
    )
    rows[f"mae_vs_saved_reg_target_{k}"] = float(
      np.mean(np.abs(diff_reg[:, k]))
    )
  rows["status"] = "ok"
  return rows


def plot_heatmap(distances_df: pd.DataFrame, plt_dir: str, keyword: str):
  for target in ["aggregate_distance", "similarity_score"]:
    # Convert the long-form DataFrame to a matrix.
    distance_matrix = distances_df.pivot(
      index = "model_node",
      columns = "data_node",
      values = target,
    )
    # Sort nodes numerically.
    distance_matrix = distance_matrix.sort_index().sort_index(axis=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
      distance_matrix,
      annot = True,
      fmt = ".3f",
      cmap = "viridis",
      linewidths = 0.5,
      square = True,
      cbar_kws = {"label": target},
    )
    plt.title(f"Pairwise {target}")
    plt.xlabel("Dataset node")
    plt.ylabel("Model node")
    plt.tight_layout()
    plt.savefig(
      os.path.join(plt_dir, f"{target}_heatmap-{keyword}.png"), dpi=300
    )
    plt.close()


def main(
    base_dir: str, 
    results_dir: str, 
    regression_weight: float,
    nodes: Any,
    results_seed: int
  ):
  output_dir = os.path.join(results_dir, "node_distance_analysis")
  os.makedirs(output_dir, exist_ok = True)
  # Load all data first; this also validates that all requested nodes exist
  print("Loading nodes data...")
  node_data = {} # dict[int, tuple[np.ndarray, np.ndarray]]
  if str(nodes) != "all" and str(nodes) != "['all']":
    for node_id in nodes:
      data_path = os.path.join(base_dir, f"{node_id}_data.json")
      if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")
      node_data[node_id] = load_node_data(data_path)
  else:
    nodes = []
    for fname in os.listdir(base_dir):
      if fname.endswith("_data.json") and not fname.startswith("centralized"):
        node_id = int(fname.replace("_data.json", ""))
        nodes.append(node_id)
        data_path = os.path.join(base_dir, fname)
        node_data[node_id] = load_node_data(data_path)
  print(f"    DONE! Found nodes: {nodes}")
  # Load each model once
  print("Loading models...")
  models = {}
  for node_id in nodes:
    model_path = os.path.join(results_dir, f"{node_id}_single.keras")
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Missing model file: {model_path}")
    # compile=False avoids needing custom training losses just for inference.
    models[node_id] = load_model(model_path, compile = False)
  print("    DONE!")
  # Compute predictions and distance
  print("Computing predictions and distance...")
  distances_path = os.path.join(output_dir, "pairwise_distances.csv")
  prediction_distances_path = os.path.join(
    output_dir, "pairwise_prediction_distances.csv"
  )
  distances_df = pd.DataFrame()
  prediction_distances_df = pd.DataFrame()
  if os.path.exists(distances_path) and \
        os.path.exists(prediction_distances_path):
    distances_df = pd.read_csv(distances_path)
    prediction_distances_df = pd.read_csv(prediction_distances_path)
  else:
    distance_rows = []
    prediction_distance_rows = []
    # Cache each model's own-data, node_type=-1 predictions
    own_minus_one_predictions = {}
    for node_id in nodes:
      X, _ = node_data[node_id]
      pair_seed = results_seed#args.seed + node_id * 1009 + node_id * 9176
      own_minus_one_predictions[node_id] = predict(
        models[node_id], replace_node_type_with_minus_one(X), 6, pair_seed
      )
    # Load saved real-node-type regression predictions once per node (reference 1).
    saved_real_type_predictions = {}
    saved_prediction_paths = {}
    for node_id in nodes:
      pred_path = os.path.join(results_dir, f"{node_id}_single_pred.json")
      saved_prediction_paths[node_id] = pred_path
      if os.path.exists(pred_path):
        saved_real_type_predictions[node_id] = load_saved_predictions(
          pred_path, node_id, "train"
        )
    for data_node in nodes:
      X, Y = node_data[data_node]
      X_minus_one = replace_node_type_with_minus_one(X)
      for model_node in nodes:
        model = models[model_node]
        # # Use a stable but distinct seed per model/data pair.
        # pair_seed = args.seed + model_node * 1009 + data_node * 9176
        pair_seed = results_seed
        reg_pred, cls_prob = predict(model, X_minus_one, 6, pair_seed)
        metrics = calculate_distances(Y, reg_pred, cls_prob, regression_weight)
        distance_rows.append(
            {"model_node": model_node, "data_node": data_node, **metrics}
        )
        # Compare to data_node's saved predictions made with its real node type
        saved = saved_real_type_predictions.get(data_node)
        if saved is not None:
          saved_metrics = calculate_distances(
            np.concatenate((saved[0], saved[2]), axis=1), 
            reg_pred, 
            cls_prob, 
            regression_weight
          )
          prediction_distance_rows.append({
            "reference": "saved_real_node_type",
            "model_node": model_node,
            "data_node": data_node,
            "reference_node": data_node,
            "reference_file": str(saved_prediction_paths[data_node]),
            **saved_metrics,
          })
        else:
          prediction_distance_rows.append({
            "reference": "saved_real_node_type",
            "model_node": model_node,
            "data_node": data_node,
            "reference_node": data_node,
            "reference_file": str(saved_prediction_paths[data_node]),
            "status": "missing saved prediction file",
          })
        # Compare to data_node's own model on the same data with node_type=-1.
        ref_minus_one = own_minus_one_predictions[data_node]
        minus_one_metrics = calculate_distances(
          np.concatenate(
            (ref_minus_one[0], ref_minus_one[1].reshape((-1,1))), axis=1
          ), 
          reg_pred, 
          cls_prob, 
          regression_weight
        )
        prediction_distance_rows.append({
          "reference": "own_model_node_type_minus_one",
          "model_node": model_node,
          "data_node": data_node,
          "reference_node": data_node,
          "reference_file": "computed in this run",
          **minus_one_metrics,
        })
      print(f"    Finished {len(nodes)} models on data {data_node}")
    distances_df = pd.DataFrame(distance_rows)
    distances_df.to_csv(distances_path, index=False)
    prediction_distances_df = pd.DataFrame(prediction_distance_rows)
    prediction_distances_df.to_csv(prediction_distances_path, index=False)
  # The diagonal entries are also available in pairwise_distances.csv. This
  # separate diagnostic explicitly compares node_type=-1 predictions against
  # the previously saved predictions (normally made with the original type).
  diagnostic_df = pd.DataFrame()
  diagnostic_path = os.path.join(output_dir, "own_data_minus_one_vs_saved.csv")
  if os.path.exists(diagnostic_path):
    diagnostic_df = pd.read_csv(diagnostic_path)
  else:
    diagnostic_rows = []
    for node_id in nodes:
        pred_path = os.path.join(results_dir, f"{node_id}_single_pred.json")
        if not os.path.exists(pred_path):
            diagnostic_rows.append({
            "node": node_id, 
            "status": f"missing saved predictions: {pred_path}"
            })
            continue
        X, _ = node_data[node_id]
        row = run_diagnostic(
            node_id = node_id,
            X = X,
            model = models[node_id],
            saved_predictions_path = pred_path,
            num_permutations = 6,
            seed = results_seed#args.seed + node_id * 1009 + node_id * 9176,
        )
        diagnostic_rows.append(row)
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    diagnostic_df.to_csv(diagnostic_path, index=False)
  print("\nDone.")
  print(f"Pairwise distances: {distances_path}")
  print(f"Own-data -1 diagnostic: {diagnostic_path}")
  # Plot
  plot_heatmap(distances_df, output_dir, "vs_ground_truth")
  plot_heatmap(
    prediction_distances_df[
      prediction_distances_df["reference"]=="saved_real_node_type"
    ], 
    output_dir,
    "saved_real_node_type"
  )
  plot_heatmap(
    prediction_distances_df[
      prediction_distances_df["reference"]=="own_model_node_type_minus_one"
    ], 
    output_dir,
    "own_model_node_type_minus_one"
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_dir = args.base_dir
  results_seed = args.results_seed
  results_dir = os.path.join(base_dir, f"results_{results_seed}")
  regression_weight = args.regression_weight
  if not 0.0 <= regression_weight <= 1.0:
    raise ValueError("regression-weight must be between 0 and 1.")
  nodes = args.nodes
  main(base_dir, results_dir, regression_weight, nodes, results_seed)
