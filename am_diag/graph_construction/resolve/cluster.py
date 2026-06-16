"""Balanced K-Means clustering for embedding-based item grouping."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from am_diag.common.data_models.cluster import ItemCluster


def cluster_items(
    embeddings: np.ndarray,
    cluster_size: int,
    max_iter: int = 20,
) -> list[ItemCluster]:
    """Balanced K-Means clustering — capacity-capped bins.

    Args:
        embeddings: ``(n, d)`` float array.
        cluster_size: Target items per cluster.
        max_iter: K-Means max iterations.

    Returns:
        List of ``ItemCluster`` s.
    """
    n = len(embeddings)
    num_clusters = max(1, n // cluster_size)
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="random",
        n_init=1,
        max_iter=max_iter,
        tol=0.0,
        algorithm="lloyd",
    )
    kmeans.fit(embeddings)
    distances = cdist(embeddings, kmeans.cluster_centers_)
    assignments = np.argsort(distances, axis=1)
    buckets: list[list[int]] = [[] for _ in range(num_clusters)]
    assigned = np.zeros(n, dtype=bool)
    for rank in range(num_clusters):
        for i in range(n):
            if assigned[i]:
                continue
            cluster_id = assignments[i, rank]
            if len(buckets[cluster_id]) < cluster_size:
                buckets[cluster_id].append(i)
                assigned[i] = True
    unassigned = np.where(~assigned)[0]
    if len(unassigned):
        buckets.append(unassigned.tolist())
    return [ItemCluster(items=indices) for indices in buckets]


__all__ = ["cluster_items"]
