import numpy as np

from bidirectional_inverse.graphx import draw_q_graph_from_matrix


def test_draw_q_graph_saves(tmp_path):
    m = np.array([[1.0, -0.2], [0.1, 1.0]])
    out = tmp_path / "q_graphx.png"
    draw_q_graph_from_matrix(m, output_path=out)
    assert out.exists()
