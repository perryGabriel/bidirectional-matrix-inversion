from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from queue import PriorityQueue
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


def queue_estimate_dict(
    matrix: SparseMatrix,
    j: int,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    verbose: int = 0,
    **kwargs,
):
    del kwargs
    u: SparseVector = {}
    queue: PriorityQueue[tuple[float, bool, int, int]] = PriorityQueue()
    queue.put((-1.0, False, j, 0))
    flops = 0
    cols: set[int] = set()

    for _ in range(max_iterations):
        if queue.empty():
            break
        neg_flow_mag, is_neg, col, depth = queue.get()
        del depth
        flops += 1
        signed_flow = -neg_flow_mag * ((-1) ** int(is_neg))
        u[col] = u.get(col, 0.0) + signed_flow
        cols.add(col)

        if -neg_flow_mag > epsilon and col in matrix:
            for row in matrix[col].keys():
                q = get_q_entry(matrix, row, col)
                flops += 1
                queue.put((abs(q) * neg_flow_mag, is_neg ^ (q < 0), row, 0))
    if verbose > 0:
        print(f"queue size at end: {queue.qsize()}")
    return u, flops, len(cols)


def recover_power_series_dict(
    matrix: SparseMatrix,
    j: int,
    u: SparseVector | None = None,
    curr: SparseVector | None = None,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    verbose: int = 0,
    **kwargs,
):
    del kwargs
    if u is None:
        u = {}
    if curr is None:
        curr = {j: 1.0}
    remainder: SparseVector = {}
    flops = 0
    cols: set[int] = set()

    for iteration in range(max_iterations):
        u, local = add_vectors(u, curr)
        flops += local
        curr, local = mul_q_left(matrix, curr)
        flops += local
        cols |= set(curr.keys())
        magnitude = measure_dict_vector(curr)
        if verbose > 0 and iteration % 5 == 0:
            print(f"||Q^k e_{j}|| = {magnitude:.4f}")
        if magnitude < epsilon:
            break

        for key in list(curr.keys()):
            if abs(curr[key]) < epsilon:
                remainder[key] = remainder.get(key, 0.0) + curr[key]
                del curr[key]

    for key, value in remainder.items():
        u[key] = u.get(key, 0.0) + value
    return u, flops, len(cols)


def ml_estimate_dict(
    matrix: SparseMatrix,
    j: int,
    max_iterations: int = 10,
    epsilon: float = 1e-3,
    learning_rate: float = 1.0,
    verbose: int = 0,
    **kwargs,
):
    del kwargs, verbose
    u: SparseVector = {j: 1.0}
    flops = 0
    cols: set[int] = set()
    for _ in range(max_iterations):
        y, local = dict_mat_mult(matrix, u)
        flops += local
        scale = y.get(j, 1.0)
        if abs(scale) <= epsilon:
            scale = 1.0

        for row in list(u.keys()):
            u[row] = u[row] / scale
            flops += 1
            cols.add(row)

        stop_early = True
        for row in list(y.keys()):
            if abs((row == j) - y[row]) > epsilon:
                stop_early = False
            if row not in u:
                u[row] = 0.0
            if row != j and row in matrix and row in matrix[row] and abs(matrix[row][row]) > epsilon:
                u[row] -= learning_rate * y[row] / scale / matrix[row][row]
                flops += 1
            if abs(u[row]) < epsilon:
                del u[row]
        if stop_early:
            break
    return u, flops, len(cols)


@dataclass
class TreeNode:
    pos: int
    parent: "TreeNode | None" = None
    value: float = 0.0
    children_nodes: dict[int, "TreeNode"] | None = None
    children_edges: dict[int, float] | None = None

    def __post_init__(self):
        if self.children_nodes is None:
            self.children_nodes = {}
        if self.children_edges is None:
            self.children_edges = {}

    def get_child(self, q: np.ndarray, child_index: int):
        if child_index not in self.children_nodes:
            self.children_nodes[child_index] = TreeNode(pos=child_index, parent=self)
            self.children_edges[child_index] = float(q[child_index, self.pos])
        return self.children_nodes[child_index], self.children_edges[child_index]

    def update_value(self, target: int):
        self.value = float(sum(self.children_nodes[j].value * self.children_edges[j] for j in self.children_nodes))
        if self.pos == target:
            self.value += 1.0

    def rollout(self, q: np.ndarray, target: int, curr_flow: float, epsilon: float):
        candidates = np.nonzero(q[:, self.pos])[0]
        if len(candidates) == 0:
            return
        child_idx = int(np.random.choice(candidates))
        child, edge = self.get_child(q, child_idx)
        new_flow = curr_flow * edge
        if abs(new_flow) > epsilon:
            child.rollout(q, target, new_flow, epsilon)
        self.update_value(target)


def monte_carlo_estimate_entry(
    q: np.ndarray,
    row: int,
    col: int,
    epsilon: float = 1e-6,
    num_iters: int = int(1e4),
):
    root = TreeNode(pos=col, parent=None)
    for _ in range(num_iters):
        root.rollout(q, target=row, curr_flow=1.0, epsilon=epsilon)
    return root.value
