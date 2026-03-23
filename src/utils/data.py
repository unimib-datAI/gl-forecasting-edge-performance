from gossiplearning.models import LabelledData, Dataset, NodeId

from sklearn.model_selection import train_test_split
from typing import Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import random
import json
import os



def encode_sequences_for_training(
    df: pd.DataFrame,
    input_steps: int,
    output_steps: int,
    n_output_vars: int,
    n_auxiliary_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    res = pd.concat(
        [df.shift(-i) for i in range(0, input_steps + output_steps)], axis=1
    )

    res = res.dropna(axis=0)

    n_timestep_features = n_output_vars + n_auxiliary_features
    n_input_cols = input_steps * n_timestep_features

    X, Y = (
        res.iloc[:, :n_input_cols].to_numpy(),
        res.iloc[:, n_input_cols : (n_input_cols + n_output_vars)].to_numpy(),
    )
    X = X.reshape((len(X), input_steps, n_timestep_features))
    Y = Y.reshape((len(X), n_output_vars))

    return X, Y


def train_val_test_split(
    X: np.ndarray, Y: np.ndarray, test_perc: float, val_perc_on_train: float
) -> tuple[np.ndarray, ...]:
    train_len = int(len(X) * (1 - test_perc))

    X_tmp, X_test = X[:train_len], X[train_len:]
    Y_tmp, Y_test = Y[:train_len], Y[train_len:]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tmp, Y_tmp, test_size=val_perc_on_train, shuffle=False
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_dataset(
    data_folder: str, dataset_key: str
  ) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
  ]:
  X_train, Y_train, X_val, Y_val, X_test, Y_test = [None] * 6
  # open file
  filename = os.path.join(data_folder, f"{dataset_key}_data.json")
  with open(filename, "r") as ist:
    # load data
    data_json = json.load(ist)
    X_train = np.array(data_json["X_train"])
    Y_train = np.array(data_json["Y_train"])
    X_val = np.array(data_json["X_val"])
    Y_val = np.array(data_json["Y_val"])
    X_test = np.array(data_json["X_test"])
    Y_test = np.array(data_json["Y_test"])
  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def load_tasks(data_folder: str, unified: bool = False) -> dict:
    tasks = {}
    prefix = "unified_" if unified else ""
    filename = os.path.join(data_folder, f"{prefix}tasks.json")
    with open(filename, "r") as istream:
       tasks = json.load(istream)
    return tasks


def save_dataset(
    X_train: np.ndarray, 
    Y_train: np.ndarray, 
    X_val: np.ndarray, 
    Y_val: np.ndarray, 
    X_test: np.ndarray, 
    Y_test: np.ndarray, 
    output_folder: str,
    dataset_key: str
  ):
  # convert data to json
  data_json = {
    "X_train": X_train.tolist(), 
    "Y_train": Y_train.tolist(), 
    "X_val": X_val.tolist(), 
    "Y_val": Y_val.tolist(), 
    "X_test": X_test.tolist(), 
    "Y_test": Y_test.tolist()
  }
  # save
  filename = os.path.join(output_folder, f"{dataset_key}_data.json")
  with open(filename, "w") as ost:
    ost.write(json.dumps(data_json, indent = 2))


def load_npz_data(file: str) -> tuple[np.ndarray, ...]:
    data = np.load(file)
    return (
        data["X_train"],
        data["Y_train"],
        data["X_val"],
        data["Y_val"],
        data["X_test"],
        data["Y_test"],
    )


def get_test_sets(base_dir: Path, n_nodes: int) -> tuple[LabelledData]:
    result = []
    for i in range(n_nodes):
        _, _, _, _, X_test, Y_test = load_npz_data(str(base_dir / f"node_{i}.npz"))
        result.append((X_test, Y_test))

    return tuple(result)


def prepare_history_for_training(
    history: pd.DataFrame,
    input_steps: int,
    output_steps: int,
    n_output_vars: int,
    n_auxiliary_features: int,
) -> tuple[np.ndarray, ...]:
    X, Y = encode_sequences_for_training(
        history, input_steps, output_steps, n_output_vars, n_auxiliary_features
    )
    return train_val_test_split(X, Y, test_perc=0.3, val_perc_on_train=0.1)


def prepare_dataset_for_training(
    towers_datasets: list[pd.DataFrame],
    output_folder: Path,
    input_timesteps: int,
    output_timesteps: int,
    n_functions: int,
    n_auxiliary_features: int = 0,
) -> None:
    nodes_datasets = {
        i: prepare_history_for_training(
            dataset,
            input_steps=input_timesteps,
            output_steps=output_timesteps,
            n_output_vars=n_functions,
            n_auxiliary_features=n_auxiliary_features,
        )
        for i, dataset in enumerate(towers_datasets)
    }

    output_folder.mkdir(exist_ok=True, parents=True)

    for i, (X_train, Y_train, X_val, Y_val, X_test, Y_test) in nodes_datasets.items():
        np.savez(
            str(output_folder / f"node_{i}"),
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
        )

    print("Saved!\n")


def get_common_test_set(
    node_data_fn: Callable[[NodeId], Dataset], n_nodes: int, perc: float
) -> LabelledData:
    datasets = [node_data_fn(j) for j in range(n_nodes)]

    test_sets = [(ds["X_test"], ds["Y_test"]) for ds in datasets]

    test: list[LabelledData] = []

    for ts in test_sets:
        num_records = len(ts[0])
        indices = np.random.choice(
            num_records, replace=False, size=round(perc * num_records)
        )

        test.append((ts[0][indices], ts[1][indices]))

    testX = np.concatenate([t[0] for t in test])
    testY = np.concatenate([t[1] for t in test])

    return (testX, testY)

