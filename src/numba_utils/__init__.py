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
    """
    distances[i, j] = (train_matrix[i] - val_matrix[j]).(train_matrix[i] - val_matrix[j])
                    = train_matrix[i].train_matrix[i]
                        + val_matrix[j].val_matrix[j]
                        - 2 * (train_matrix[i] . val_matrix[j])
    t2_diagonal = main_diagonal_of(train_matrix . train_matrix.T)
    v2_diagonal = main_diagonal_of(val_matrix . val_matrix.T)
    train_dot_val = train_matrix . val_matrix
    distances[i, j] = t2_diagonal[i] + v2_diagonal[j] - 2 * train_dot_val[i][j]
    """
    t2_diagonal = np.sum(np.square(train_matrix), axis=1)
    v2_diagonal = np.sum(np.square(val_matrix), axis=1)
    train_dot_val = np.dot(train_matrix, val_matrix.T)
    distances = np.zeros((train_matrix.shape[0], val_matrix.shape[0]), dtype=np.float32)
    for i in range(len(t2_diagonal)):
        for j in range(len(v2_diagonal)):
            distances[i][j] = np.sqrt(
                t2_diagonal[i] + v2_diagonal[j] - 2 * train_dot_val[i][j]
            )
    return distances
