from __future__ import annotations

import numpy as np


class TreeNode:
    def __init__(self, pos: int, parent: "TreeNode | None"):
        self.pos = pos
        self.parent = parent
        self.children_nodes: dict[int, TreeNode] = {}
        self.children_edges: dict[int, float] = {}
        self.value = 0.0

    def get_child(self, q: np.ndarray, child_index: int) -> tuple["TreeNode", float]:
        if child_index not in self.children_nodes:
            self.children_nodes[child_index] = TreeNode(child_index, self)
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


def estimate_entry(q: np.ndarray, row: int, col: int, epsilon: float = 1e-6, num_iters: int = int(1e4)) -> float:
    root = TreeNode(pos=col, parent=None)
    for _ in range(num_iters):
        root.rollout(q, row, curr_flow=1.0, epsilon=epsilon)
    return root.value
