from tensorflow.keras.utils import Sequence

from itertools import permutations
import random
import numpy as np


class JanossyPermutedBatchSequence(Sequence):
    """
    Keras Sequence generator for Janossy pooling.
    Generates `k` permutations per sample in each batch.
    """

    def __init__(self, X, y, batch_size=32, num_permutations=10, shuffle_batch=True):
        self.X = np.array(X)  # shape: (num_samples, sequence_length, input_dim)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.num_permutations = num_permutations
        self.shuffle_batch = shuffle_batch
        self.indexes = np.arange(len(self.X))
        self.seq_len = self.X.shape[1]

        # Precompute all permutations if sequence length is small
        if self.seq_len <= 6:
            self.all_perms = list(permutations(range(self.seq_len)))
        else:
            self.all_perms = None  # fallback to random sampling

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]

        # Output shape: (batch_size, num_permutations, sequence_length, input_dim)
        X_batch_permuted = np.zeros((len(batch_indexes), self.num_permutations, self.seq_len, self.X.shape[2]))

        for i, sample in enumerate(X_batch):
            if self.all_perms and len(self.all_perms) >= self.num_permutations:
                selected_perms = random.sample(self.all_perms, self.num_permutations)
            else:
                selected_perms = [np.random.permutation(self.seq_len) for _ in range(self.num_permutations)]

            for j, perm in enumerate(selected_perms):
                X_batch_permuted[i, j] = sample[list(perm)]

        return X_batch_permuted, y_batch

    def on_epoch_end(self):
        if self.shuffle_batch:
            np.random.shuffle(self.indexes)


def prepare_janossy_input(X, Y, num_permutations=10):
  """
  Transforms test data to Janossy-pooling-compatible input shape:
  (num_samples, num_permutations, sequence_length, input_dim)

  Args:
      X: shape (num_samples, sequence_length, input_dim)
      num_permutations: number of permutations to generate per sample
  Returns:
      X_prepared: shape (num_samples, num_permutations, sequence_length, input_dim)
  """
  # X
  X_prepared = prepare_janossy_test_input(X, num_permutations)
  # Y
  Y_reg, Y_cls = Y[:, :-1], Y[:, -1]
  Y_prepared = {"fn_0": Y_reg, "fn_1": Y_cls}
  return X_prepared, Y_prepared


def prepare_janossy_test_input(X, num_permutations=10):
  """
  Transforms test data to Janossy-pooling-compatible input shape:
  (num_samples, num_permutations, sequence_length, input_dim)

  Args:
      X: shape (num_samples, sequence_length, input_dim)
      num_permutations: number of permutations to generate per sample
  Returns:
      X_prepared: shape (num_samples, num_permutations, sequence_length, input_dim)
  """
  X = np.array(X)
  num_samples, seq_len, feat_dim = X.shape
  # -- try to precompute all permutations if small enough
  if seq_len <= 6:
    all_perms = list(permutations(range(seq_len)))
  else:
    all_perms = None
  X_prepared = np.zeros((num_samples, num_permutations, seq_len, feat_dim))
  for i in range(num_samples):
    if all_perms and len(all_perms) >= num_permutations:
      selected_perms = random.sample(all_perms, num_permutations)
    else:
      selected_perms = [np.random.permutation(seq_len) for _ in range(num_permutations)]
    for j, perm in enumerate(selected_perms):
      X_prepared[i, j] = X[i, list(perm)]
  return X_prepared

