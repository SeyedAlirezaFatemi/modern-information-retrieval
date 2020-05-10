import numba as nb
import numpy as np

from numba import njit


@njit(
    nb.float32[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64, nb.int64),
    parallel=True,
    fastmath=True,
)
def k_means(features, initial_centroids, num_centers, num_iterations):
    centroids = initial_centroids
    num_samples, num_features = features.shape
    dist = np.zeros((num_samples, num_centers), dtype=np.float32)
    for _ in range(num_iterations):

        for i in range(num_samples):
            for j in range(num_centers):
                dist[i, j] = np.sqrt(np.sum((features[i, :] - centroids[j, :]) ** 2))

        labels = np.array([dist[i, :].argmin() for i in range(num_samples)])

        for i in range(num_centers):
            centroids[i, :] = np.sum(features[labels == i, :], axis=0) / np.sum(
                labels == i
            )

    return centroids
