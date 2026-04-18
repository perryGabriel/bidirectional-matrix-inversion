import numpy as np

from bidirectional_inverse.algorithms import (
    TreeNode,
    add_vectors,
    dict_mat_mult,
    gaussian_inv_dict,
    ml_estimate_dict,
    monte_carlo_estimate_entry,
    mul_q_left,
    pow_estimate_dict,
    pow_estimate_epsilon_dict,
    queue_estimate_dict,
    recover_power_series_dict,
)
from bidirectional_inverse.bidirectional import bidir_dict
from bidirectional_inverse.matrix import dict_to_matrix, generate_sparse_adjacency_list


def _small_matrix(seed=4):
    m_row, m_col = generate_sparse_adjacency_list(6, 2, 0.5, seed=seed)
    return m_row, m_col


def test_add_vectors_and_mult():
    out, nnz = add_vectors({0: 1.0}, {0: 2.0, 2: 3.0})
    assert nnz == 2
    assert out == {0: 3.0, 2: 3.0}

    m = {0: {0: 2.0, 1: 1.0}, 1: {1: 3.0}}
    y, flops = dict_mat_mult(m, {0: 1.0, 1: 2.0})
    assert flops > 0
    assert np.isclose(y[0], 2.0)


def test_mul_q_left_runs():
    _, m_col = _small_matrix()
    y, flops = mul_q_left(m_col, {0: 1.0}, epsilon=1e-9)
    assert flops > 0
    assert isinstance(y, dict)


def test_gaussian_inverse_column_close_to_numpy():
    _, m_col = _small_matrix(seed=7)
    dense = dict_to_matrix(m_col, 6)
    exact = np.linalg.inv(dense)
    est, _, _ = gaussian_inv_dict(m_col, j=2)
    for r, val in est.items():
        assert np.isclose(val, exact[r, 2], atol=1e-5)


def test_power_and_priority_estimates_reasonable():
    _, m_col = _small_matrix(seed=8)
    dense = dict_to_matrix(m_col, 6)
    exact_col = np.linalg.inv(dense)[:, 1]

    est_pow, _, _ = pow_estimate_dict(m_col, j=1, max_iterations=200, epsilon=1e-7)
    est_pri, _, _ = pow_estimate_epsilon_dict(m_col, j=1, max_iterations=200, epsilon=1e-7)

    pow_vec = np.zeros(6)
    pri_vec = np.zeros(6)
    for k, v in est_pow.items():
        pow_vec[k] = v
    for k, v in est_pri.items():
        pri_vec[k] = v

    assert np.linalg.norm(pow_vec - exact_col, ord=np.inf) < 5e-2
    assert np.linalg.norm(pri_vec - exact_col, ord=np.inf) < 5e-2


def test_bidir_runs_and_returns_sparse_vector():
    m_row, m_col = _small_matrix(seed=9)
    u, flops, cols = bidir_dict(m_col, j=1, matrix_row=m_row, i=2, max_iterations=100, epsilon=1e-5)
    assert isinstance(u, dict)
    assert flops > 0
    assert cols >= 0


def test_queue_recover_ml_algorithms_smoke():
    _, m_col = _small_matrix(seed=11)
    for method in (queue_estimate_dict, recover_power_series_dict, ml_estimate_dict):
        u, flops, cols = method(m_col, j=1, max_iterations=30, epsilon=1e-5)
        assert isinstance(u, dict)
        assert flops >= 0
        assert cols >= 0


def test_monte_carlo_entry_smoke():
    q = np.array([[0.0, 0.1], [0.2, 0.0]])
    val = monte_carlo_estimate_entry(q, row=0, col=1, num_iters=20)
    assert isinstance(val, float)


def test_tree_node_init():
    node = TreeNode(pos=1)
    assert node.pos == 1
