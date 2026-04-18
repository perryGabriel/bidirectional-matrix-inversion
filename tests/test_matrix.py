import numpy as np

from bidirectional_inverse.matrix import (
    dict_to_matrix,
    dict_to_vector,
    generate_sparse_adjacency_list,
    get_q_entry,
    measure,
    measure_dict_matrix,
    measure_dict_vector,
)


def test_measure_variants():
    arr = np.array([1.0, -2.0, 0.0])
    assert measure(arr, 0) == 2
    assert measure(arr, 1) == 3.0
    assert np.isclose(measure(arr, 2), np.sqrt(5))
    assert measure(arr, np.inf) == 2.0


def test_dict_conversions():
    vect = {0: 1.0, 2: -2.0}
    dense = dict_to_vector(vect, 4)
    assert dense.tolist() == [1.0, 0.0, -2.0, 0.0]

    matrix = {0: {0: 1.0}, 2: {1: 5.0}}
    dense_m = dict_to_matrix(matrix, 3)
    assert dense_m[0, 0] == 1.0
    assert dense_m[1, 2] == 5.0


def test_generate_sparse_adjacency_list_shapes_and_q_entry():
    m_row, m_col = generate_sparse_adjacency_list(8, 3, 0.7, seed=1)
    assert len(m_row) == 8
    assert len(m_col) == 8
    q00 = get_q_entry(m_col, 0, 0)
    assert isinstance(q00, float)


def test_measure_dict_helpers():
    mat = {0: {0: 1.0, 2: 2.0}, 1: {1: -3.0}}
    assert measure_dict_vector(mat[0], 1) == 3.0
    assert measure_dict_matrix(mat, 1) == 6.0
