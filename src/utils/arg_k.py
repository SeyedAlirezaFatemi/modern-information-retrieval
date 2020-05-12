import numpy as np


def arg_k(matrix: np.ndarray, k: int, find_max: bool = True):
    if find_max:
        return np.argpartition(matrix, -k, axis=0)[-k:, :]
    return np.argpartition(matrix, k, axis=0)[-k:, :]
