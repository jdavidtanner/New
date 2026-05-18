"""kNN graph construction for Ricci curvature."""

from __future__ import annotations

import numpy as np

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False


def build_knn_graph(X: np.ndarray, k: int = 10) -> "nx.Graph | None":
    if not _NX:
        return None
    import networkx as nx
    n = X.shape[0]
    sq = (X ** 2).sum(axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
    np.fill_diagonal(dists, np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        nn = np.argsort(dists[i])[:k]
        for j in nn:
            G.add_edge(int(i), int(j), weight=float(dists[i, j]))
    return G


def adjacency_matrix(G: "nx.Graph", n: int) -> np.ndarray:
    if not _NX:
        return np.zeros((n, n))
    import networkx as nx
    return nx.to_numpy_array(G, nodelist=range(n))
