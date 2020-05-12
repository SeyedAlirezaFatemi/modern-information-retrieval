import numpy as np


def cosine_normalize(mat: np.ndarray):
    return mat / np.linalg.norm(mat, axis=1)[:, np.newaxis]
