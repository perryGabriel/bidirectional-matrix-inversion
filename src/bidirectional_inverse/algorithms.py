from __future__ import annotations

import copy
from collections import defaultdict
from typing import Callable

import numpy as np

from .matrix import get_q_entry, measure, measure_dict_vector
from .types import SparseMatrix, SparseVector

Method = Callable[..., tuple[SparseVector, float, int]]


def add_vectors(left: SparseVector, right: SparseVector) -> tuple[SparseVector, int]:
    for row, value in right.items():
        left[row] = left.get(row, 0.0) + value
    return left, len(right)


def dict_mat_mult(matrix: SparseMatrix, vector: SparseVector, epsilon: float = 0.0):
    y: SparseVector = defaultdict(float)
    flops = 0
    for col in vector.keys() & matrix.keys():
        flops += 2 * len(matrix[col])
        for row, val in matrix[col].items():
            y[row] += val * vector[col]
            if abs(y[row]) <= epsilon:
                del y[row]
    return dict(y), flops


def mul_q_left(matrix: SparseMatrix, vector: SparseVector, epsilon: float = 0.0):
    y: SparseVector = defaultdict(float)
    flops = 0
    for col, vcol in vector.items():
        if col not in matrix:
            continue
        flops += 2 * len(matrix[col])
        for row in matrix[col].keys():
            y[row] += get_q_entry(matrix, row, col) * vcol
            if abs(y[row]) < epsilon:
                del y[row]
    return dict(y), flops


def gaussian_inv_dict(matrix: SparseMatrix, j: int, verbose: int = 0, **kwargs):
    del verbose, kwargs
    m = copy.deepcopy(matrix)
    size = len(m.keys())
    m[size] = {j: 1.0}
    flops = 0

    for pivot in range(size):
        max_el = 0.0
        max_row = pivot
        for row in m[pivot].keys():
            if row < pivot:
                continue
            if abs(m[pivot][row]) > max_el:
                max_el = abs(m[pivot][row])
                max_row = row

        for col in range(pivot, size + 1):
            temp = m[col].get(max_row, 0.0)
            if pivot in m[col]:
                m[col][max_row] = m[col][pivot]
            else:
                m[col].pop(max_row, None)
            if temp != 0.0:
                m[col][pivot] = temp
            else:
                m[col].pop(pivot, None)

        pivot_val = m[pivot].get(pivot, 0.0)
        if pivot_val == 0.0:
            continue

        for col in range(size, pivot - 1, -1):
            if pivot in m[col]:
                m[col][pivot] /= pivot_val
                flops += 1

        for row in list(m[pivot].keys()):
            if row == pivot:
                continue
            leading = m[pivot][row]
            for col in range(size, pivot - 1, -1):
                if row not in m[col]:
                    m[col][row] = 0.0
                if pivot in m[col]:
                    m[col][row] -= leading * m[col][pivot]
                    flops += 2

    return m[size], flops, len(matrix)


def pow_estimate_dict(
    matrix: SparseMatrix,
    j: int,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    verbose: int = 0,
    **kwargs,
):
    del kwargs
    u: SparseVector = {}
    curr: SparseVector = {j: 1.0}
    flops = 0
    cols: set[int] = set()

    for iteration in range(max_iterations):
        u, local = add_vectors(u, curr)
        flops += local

        curr, local = mul_q_left(matrix, curr)
        flops += local
        cols |= set(curr.keys())

        magnitude = measure(list(curr.values()))
        if verbose > 0 and iteration % 5 == 0:
            print(f"||Q^k e_{j}|| = {magnitude:.4f}")
        if magnitude < epsilon:
            break

    return u, flops, len(cols)


def pow_estimate_epsilon_dict(
    matrix: SparseMatrix,
    j: int,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    verbose: int = 0,
    **kwargs,
):
    del kwargs
    u: SparseVector = {}
    curr: SparseVector = {j: 1.0}
    flops = 0
    cols: set[int] = set()

    for iteration in range(max_iterations):
        u, local = add_vectors(u, curr)
        flops += local

        curr, local = mul_q_left(matrix, curr, epsilon=epsilon)
        flops += local
        cols |= set(curr.keys())

        magnitude = measure_dict_vector(curr)
        if verbose > 0 and iteration % 5 == 0:
            print(f"||Q^k e_{j}|| = {magnitude:.4f}")
        if magnitude < epsilon:
            break

    return u, flops, len(cols)


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
