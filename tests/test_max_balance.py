import contexts as ctx

import numpy as np
from scipy.sparse import isspmatrix_csr

from snpp.cores.max_balance import greedy, greedy_g, \
    faster_greedy, \
    edge_weight_sum
from snpp.cores.triangle import build_edge2edges
from test_triangle import A6, g6


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


C = np.array(['a', 'b', 'c', 'c', 'd', 'x'])
targets = set([(0, 1), (4, 5)])


# faster_greedy
def test_faster_greedy_1(g6):
    preds = faster_greedy(g6, C, B=1, T=targets)
    assert preds == [(0, 1, -1)]
    assert not g6.has_edge(0, 1)


def test_faster_greedy_2(g6): 
    preds = faster_greedy(g6, C, B=2, T=targets)
    assert preds == [(0, 1, -1), (4, 5, 1)]
    assert not g6.has_edge(4, 5)  # no side effect


def test_faster_greedy_3(g6):
    preds = faster_greedy(g6, C, B=100, T=targets)
    assert preds == [(0, 1, -1), (4, 5, 1)]
    assert not g6.has_edge(4, 5)


def test_faster_greedy_4(g6):
    preds = faster_greedy(g6, C, B=10,
                          T={(0, 1), (1, 5), (2, 5), (2, 3), (3, 5)})
    assert preds == [(2, 3, 1), (0, 1, -1), (2, 5, -1), (1, 5, -1), (3, 5, -1)]
    assert not g6.has_edge(4, 5)
    assert not g6.has_edge(0, 1)
    
