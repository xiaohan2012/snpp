import contexts as ctx

import numpy as np
from scipy.sparse import isspmatrix_csr

from snpp.cores.max_balance import greedy, \
    edge_weight_sum
from test_triangle import A6


def test_edge_weight_sum():
    s = edge_weight_sum([(0, 1), (1, 2)],
                        W=np.ones((3, 3)))
    assert s == 2

    s = edge_weight_sum(list(range(3)), W=None)
    assert s == 3

    
def test_greedy(A6):
    W = np.ones(A6.shape)
    C = np.array(['a', 'b', 'c', 'c', 'd', 'x'])
    targets = set([(0, 1), (4, 5)])

    P = greedy(A6, W, C, B=1, T=targets)
    assert list(zip(*P.nonzero())) == [(0, 1)]
    assert P[0, 1] == -1
    assert isspmatrix_csr(P)

    P = greedy(A6, W, C, B=2, T=targets)
    assert list(zip(*P.nonzero())) == [(0, 1), (4, 5)]
    assert P[0, 1] == -1
    assert P[4, 5] == 1

    P = greedy(A6, W, C, B=100, T=targets)
    assert list(zip(*P.nonzero())) == [(0, 1), (4, 5)]
    assert P[0, 1] == -1
    assert P[4, 5] == 1
