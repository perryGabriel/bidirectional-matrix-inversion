from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import SparseMatrix, SparseVector


def measure(arr, norm: float = 1):
    if np.size(arr) == 0:
        return 0.0
    if norm == 0:
        return int(np.count_nonzero(arr))
    if norm == 1:
        return float(np.sum(np.abs(arr)))
    if norm == 2:
        return float(np.sqrt(np.sum(np.square(arr))))
    if norm == np.inf:
        return float(np.max(np.abs(arr)))
    raise ValueError(f"Unsupported norm: {norm}")


def measure_dict_vector(vect: SparseVector, norm: float = 1):
    return measure(list(vect.values()), norm=norm)


def measure_dict_matrix(matrix: SparseMatrix, norm: float = 1):
    if norm == 0:
        return sum(measure_dict_vector(col, norm=0) for col in matrix.values())
    return sum(measure_dict_vector(col, norm=norm) for col in matrix.values())


def dict_to_vector(vect: SparseVector, size: int) -> np.ndarray:
    array = np.zeros((size,))
    for key, value in vect.items():
        array[key] = value
    return array


def dict_to_matrix(matrix: SparseMatrix, size: int) -> np.ndarray:
    array = np.zeros((size, size))
    for key, column in matrix.items():
        array[:, key] = dict_to_vector(column, size)
    return array


def get_q_entry(matrix: SparseMatrix, row: int, col: int) -> float:
    return float((row == col) - matrix[col].get(row, 0.0))


def generate_sparse_adjacency_list(
    size: int,
    num_out_edges: int,
    sum_of_each_column: float,
    seed: int | None = None,
) -> tuple[SparseMatrix, SparseMatrix]:
    if size <= 0:
        raise ValueError("size must be positive")
    if sum_of_each_column <= 0:
        raise ValueError("sum_of_each_column must be positive")

    rng = np.random.default_rng(seed)
    matrix_col: SparseMatrix = defaultdict(dict)
    matrix_row: SparseMatrix = defaultdict(dict)

    for i in range(size):
        matrix_col[i] = {i: 1.0}
        matrix_row[i] = {i: 1.0}

    num_out_edges = min(num_out_edges, size)
    for _ in range(num_out_edges):
        perm = rng.permutation(size)
        for index in range(size):
            val = float(rng.random() * 2 - 1)
            c = int(perm[index])
            r = int(perm[(index + 1) % size])
            matrix_col[c][r] = val
            matrix_row[r][c] = val

    for col in range(size):
        magnitude = np.sum(np.abs(list(matrix_col[col].values()))) / sum_of_each_column
        for row in list(matrix_col[col].keys()):
            val = float((row == col) - matrix_col[col][row] / magnitude)
            matrix_col[col][row] = val
            matrix_row[row][col] = val

    return dict(matrix_row), dict(matrix_col)
