from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .matrix import dict_to_matrix
from .types import SparseMatrix


def draw_q_graph_from_matrix(matrix: np.ndarray, output_path: Path | None = None):
    size = matrix.shape[0]
    labels = [f"v{i}" for i in range(size)]
    q = np.eye(size) - matrix
    graph = nx.DiGraph()
    for row in range(size):
        for col in range(size):
            if q[col, row] != 0:
                graph.add_edge(labels[row], labels[col], weight=float(q[col, row]))

    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_color="lightgray", node_size=450, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, width=1.0, alpha=0.7, ax=ax)
    ax.set_title("GraphX view of Q = I - M")
    ax.set_axis_off()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    return fig


def draw_q_graph_from_sparse(matrix_col: SparseMatrix, size: int, output_path: Path | None = None):
    matrix = dict_to_matrix(matrix_col, size)
    return draw_q_graph_from_matrix(matrix, output_path=output_path)
