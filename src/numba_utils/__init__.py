import numba as nb
import numpy as np
from numba import njit


@njit(nb.float32[:, :](nb.float32[:, :], nb.float32[:, :]), fastmath=True)
def calc_similarities(normalized_train_matrix, normalized_val_matrix):
    sims = np.dot(normalized_train_matrix, normalized_val_matrix.T)
    return sims


@njit(
    nb.float32[:, :](nb.float32[:, :], nb.float32[:, :]), fastmath=True, parallel=True
)
def calc_euclidean_distance(train_matrix, val_matrix):
    distances = np.zeros((train_matrix.shape[0], val_matrix.shape[0]), dtype=np.float32)
    for i in range(val_matrix.shape[0]):
        doc_vector = val_matrix[i, :]
        distances[:, i] = np.sum(np.square(train_matrix - doc_vector), axis=1)
    return distances
