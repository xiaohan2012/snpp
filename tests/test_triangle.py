import contexts as ctx

import pytest
import numpy as np

from scipy.sparse import csr_matrix
from nose.tools import assert_raises

from snpp.cores.triangle import extract_nodes_and_signs, \
    in_different_partitions, \
    get_sign_1st_order, \
    first_order_triangles_count, \
    first_order_triangles_count_g, \
    first_order_triangles_net_count_g, \
    build_edge2edges
from snpp.utils.signed_graph import matrix2graph


def test_extract_nodes_and_signs():
    # correct case
    e = (0, 1)
    e1 = (1, 2, 1)
    e2 = (0, 2, -1)
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    assert nodes == {0, 1, 2}
    assert signs == [1, -1]

    # invalid case
    # e = (0, 3)
    # e1 = (1, 2, 1)
    # e2 = (0, 2, -1)
    # assert_raises(AssertionError, extract_nodes_and_signs, e, e1, e2)


def test_in_different_partitions():
    C1 = np.array(['a', 'b', 'c'])
    assert in_different_partitions([0, 1, 2], C1)

    C2 = np.array(['a', 'a', 'b'])
    assert not in_different_partitions([0, 1, 2], C2)


def test_get_sign_1st_order():
    C1 = np.array(['a', 'b', 'c'])
    C2 = np.array(['a', 'b', 'b'])

    def make_triangle(sign_1, sign_2):
        return (0, 1), (1, 2, sign_1), (0, 2, sign_2)

    # weak balance
    e, e1, e2 = make_triangle(-1, -1)
    assert get_sign_1st_order(e, e1, e2, C1) == -1

    # strong balance
    e, e1, e2 = make_triangle(1, -1)
    assert get_sign_1st_order(e, e1, e2, C1) == -1
    
    e, e1, e2 = make_triangle(1, -1)
    assert get_sign_1st_order(e, e1, e2, C2) == -1

    e, e1, e2 = make_triangle(-1, -1)
    assert get_sign_1st_order(e, e1, e2, C2) == 1

    e, e1, e2 = make_triangle(1, 1)
    assert get_sign_1st_order(e, e1, e2, C2) == 1


@pytest.fixture
def A6():
    A = np.zeros((6, 6))

    for i in [2, 3]:
        A[0, i] = A[i, 0] = A[1, i] = A[i, 1] = -1

    for i in [4]:
        A[0, i] = A[i, 0] = A[1, i] = A[i, 1] = 1

    for i in [5]:  # not a triangle, ignore
        A[0, i] = A[i, 0] = 1
    
    return csr_matrix(A)


@pytest.fixture
def g6():
    return matrix2graph(A6(), None, multigraph=False)


def test_first_order_triangles_count(A6):
    C1 = np.array(['a', 'b', 'b', 'c', 'd', 'x'])
    C2 = np.array(['a', 'b', 'c', 'c', 'd', 'x'])
    iters = first_order_triangles_count(A6, C1, T=[(0, 1)])
    assert set(iters) == {(0, 1, -1, 1), (0, 1, 1, 2)}

    iters = first_order_triangles_count(A6, C2, T=[(0, 1)])
    assert set(iters) == {(0, 1, -1, 2), (0, 1, 1, 1)}

    iters = first_order_triangles_count(A6, C2,
                                        T=[(0, 1), (2, 3), (4, 5)])
    assert set(iters) == {(0, 1, -1, 2), (0, 1, 1, 1),
                          (2, 3, 1, 2), (4, 5, 1, 1)}


def test_first_order_triangles_count_g(g6):
    """pass nx.Graph as parameter
    """
    C1 = np.array(['a', 'b', 'b', 'c', 'd', 'x'])
    C2 = np.array(['a', 'b', 'c', 'c', 'd', 'x'])
    iters = first_order_triangles_count_g(g6, C1, T=[(0, 1)])

    assert set(iters) == {(0, 1, -1, 1), (0, 1, 1, 2)}

    iters = first_order_triangles_count_g(g6, C2, T=[(0, 1)])
    assert set(iters) == {(0, 1, -1, 2), (0, 1, 1, 1)}

    iters = first_order_triangles_count_g(g6, C2,
                                          T=[(0, 1), (2, 3), (4, 5)])
    assert set(iters) == {(0, 1, -1, 2), (0, 1, 1, 1),
                          (2, 3, 1, 2), (4, 5, 1, 1)}


def test_first_order_triangles_net_count_g(g6):
    """pass nx.Graph as parameter
    """
    C1 = np.array(['a', 'b', 'b', 'c', 'd', 'x'])
    C2 = np.array(['a', 'b', 'c', 'c', 'd', 'x'])
    iters = first_order_triangles_net_count_g(g6, C1, T=[(0, 1)])

    assert set(iters) == {(0, 1, 1, 1, (1, 2), (('s+1', 2), ('w-1', 1)))}

    iters = first_order_triangles_net_count_g(g6, C2, T=[(0, 1)])
    assert set(iters) == {(0, 1, -1, 1, (2, 1), (('w-1', 2), ('s+1', 1)))}

    iters = first_order_triangles_net_count_g(
        g6, C2,
        T=[(0, 1), (2, 3), (4, 5)])
    assert set(iters) == {(0, 1, -1, 1, (2, 1), (('w-1', 2), ('s+1', 1))),
                          (2, 3, 1, 2, (0, 2), tuple({'s+1': 2}.items())),
                          (4, 5, 1, 1, (0, 1), tuple({'s+1': 1}.items()))}


    
def test_build_edge2edges(g6):
    e2es = build_edge2edges(g6, T={(0, 1), (1, 5), (2, 5), (2, 3), (3, 5)})
    assert dict(e2es) == {(0, 1): {(1, 5)},
                          (1, 5): {(0, 1), (3, 5), (2, 5)},
                          (2, 5): {(1, 5), (2, 3), (3, 5)},
                          (2, 3): {(2, 5), (3, 5)},
                          (3, 5): {(1, 5), (2, 3), (2, 5)}}
