"""Similarity helpers."""

from typing import Tuple

import numpy as np


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query.ndim != 1:
        raise ValueError("Query embedding must be 1D")
    if matrix.ndim != 2:
        raise ValueError("Matrix embeddings must be 2D")
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix_norm @ query_norm


def top_k(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    sorted_idx = idx[np.argsort(-scores[idx])]
    return sorted_idx, scores[sorted_idx]
