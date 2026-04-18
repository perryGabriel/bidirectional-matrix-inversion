from __future__ import annotations

import numpy as np

from .algorithms import add_vectors, mul_q_left
from .matrix import measure_dict_vector
from .types import SparseMatrix, SparseVector


def bidir_dict(
    matrix_col: SparseMatrix,
    j: int,
    matrix_row: SparseMatrix,
    i: int,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    verbose: int = 0,
    **kwargs,
):
    del kwargs
    in_horizon: SparseVector = {}
    curr: SparseVector = {i: 1.0}
    sqrt_epsilon = float(np.sqrt(epsilon))
    flops = 0
    cols: set[int] = set()

    for iteration in range(max_iterations):
        in_horizon, local = add_vectors(in_horizon, curr)
        flops += local
        curr, local = mul_q_left(matrix_row, curr, epsilon=epsilon)
        flops += local
        cols |= set(curr.keys())
        magnitude = measure_dict_vector(curr)
        if verbose > 0 and iteration % 5 == 0:
            print(f"||e_{i}^TQ^k|| = {magnitude:.4f}")
        if magnitude < sqrt_epsilon:
            break

    u: SparseVector = {}
    curr = {j: 1.0}
    for iteration in range(max_iterations):
        u, local = add_vectors(u, curr)
        flops += local
        curr, local = mul_q_left(matrix_col, curr, epsilon=epsilon)
        flops += local
        magnitude = measure_dict_vector(curr)
        if verbose > 0 and iteration % 5 == 0:
            print(f"||Q^ke_{j}|| = {magnitude:.4f}")
        if magnitude < sqrt_epsilon:
            break

    for iteration in range(max_iterations):
        for row in list(curr.keys()):
            if row not in in_horizon:
                del curr[row]
        u, local = add_vectors(u, curr)
        flops += local
        curr, local = mul_q_left(matrix_col, curr, epsilon=epsilon)
        flops += local
        cols |= set(curr.keys())
        magnitude = measure_dict_vector(curr)
        if verbose > 0 and iteration % 5 == 0:
            print(f"||Q^ke_{j}|| = {magnitude:.4f}")
        if magnitude < epsilon:
            break

    return u, flops, len(cols)
