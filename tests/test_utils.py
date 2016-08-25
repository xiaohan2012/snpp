import contexts as ctx

import numpy as np
from snpp.utils import nonzero_edges, predict_signs_using_partition
from snpp.utils.data import make_lowrank_matrix


def test_nonzero_edges():
    n = 3
    M = np.eye(n)
    assert nonzero_edges(M) == {(i, i) for i in range(n)}


def test_predict_signs_using_partition():
    c = 2  # cluster count
    s = 2  # cluster size
    C = np.random.permutation([i for j in range(s) for i in range(c)])
    
    P = predict_signs_using_partition(C, targets=None)
    true_P = make_lowrank_matrix(s, rank=c)
    np.testing.assert_almost_equal(P, true_P)
    
    P = predict_signs_using_partition(C, targets=[(0, 2), (2, 0)])
    true_P = np.array([[1, 0, -1, 0],
                       [0, 1, 0, 0],
                       [-1, 0, 1, 0],
                       [0, 0, 0, 1]])
    
    np.testing.assert_almost_equal(P, true_P)
