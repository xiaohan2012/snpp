import findspark  # this needs to be the first import
findspark.init()

import contexts as ctx
import pytest
import logging

import random
import networkx as nx
import numpy as np
from scipy import sparse

from snpp.utils.data import example_for_intuition, make_lowrank_matrix, \
    make_signed_matrix
from snpp.cores.louvain import Status
from snpp.utils.signed_graph import matrix2graph, \
    to_multigraph

from pyspark import SparkConf
from pyspark import SparkContext


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_context(request):
    """ fixture for creating a spark context
    Args:
    request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName("SparkTest"))
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir('checkpoint')  # Stackoverflow error
    request.addfinalizer(lambda: sc.stop())

    quiet_py4j()
    return sc


def module_attr(request, name, default):
    return getattr(request.module, name, default)


@pytest.fixture
def rand_lowrank_mat(request):
    N = module_attr(request, 'N', 12)
    rank = module_attr(request, 'rank', 3)
    known_edge_percentage = module_attr(request, 'known_edge_percentage', 0.4)
    random_seed = module_attr(request, 'random_seed', 123456)
    random.seed(random_seed)
    np.random.seed(random_seed)

    M, _ = example_for_intuition(
        group_size=int(N / rank),
        group_number=rank,
        known_edge_percentage=known_edge_percentage)
    return sparse.csr_matrix(M)


@pytest.fixture
def rand_lowrank_g(request):
    m = rand_lowrank_mat(request)
    return matrix2graph(m, None, None, multigraph=False)


@pytest.fixture
def true_lowrank_mat(request):
    N = module_attr(request, 'N', 12)
    rank = module_attr(request, 'rank', 3)
    known_edge_percentage = module_attr(request, 'known_edge_percentage', 0.4)
    random_seed = module_attr(request, 'random_seed', 123456)

    random.seed(random_seed)
    np.random.seed(random_seed)

    _, M = example_for_intuition(
        group_size=int(N / rank),
        group_number=rank,
        known_edge_percentage=known_edge_percentage)
    return sparse.csr_matrix(M)


@pytest.fixture
def true_lowrank_g(request):
    m = true_lowrank_mat(request)
    return matrix2graph(m, None, None, multigraph=False)


@pytest.fixture
def Q1():
    """a simple signed matrix
    """
    N = 4
    friends = [(1, 2), (3, 4)]
    enemies = [(1, 3)]
    return make_signed_matrix(N, friends, enemies)


@pytest.fixture
def g1():
    """a simple signed matrix
    """
    return matrix2graph(sparse_Q1(), multigraph=False)


@pytest.fixture
def sparse_Q1():
    return sparse.csr_matrix(Q1())


@pytest.fixture
def Q1_result():
    exp = np.array([[ 1.,  1.,  -1.,  -1.],
                    [ 1.,  1.,  -1.,  -1.],
                    [-1.,  -1.,  1.,  1.],
                    [-1.,  -1.,  1.,  1.]])
    return exp


@pytest.fixture
def g1_result():
    return matrix2graph(sparse.csr_matrix(Q1_result()), multigraph=False)

    
@pytest.fixture
def Q1_d():
    """
    directed and asymmetric
    """
    Q = Q1()
    Q[0, 1] = -1
    Q[1, 3] = 0
    Q[0, 3] = 1
    return sparse.csr_matrix(Q)


@pytest.fixture
def random_graph():
    g = nx.gnp_random_graph(10, 0.5, seed=12345, directed=True)
    return nx.adjacency_matrix(g)


@pytest.fixture
def lowrank_graph(request):
    group_size = module_attr(request, 'group_size', 10)
    rank = module_attr(request, 'rank', 4)

    m = make_lowrank_matrix(group_size, rank)
    
    g = nx.Graph()
    n_row, n_col = m.shape
    for i in range(n_row):
        for j in range(n_col):
            g.add_edge(i, j, weight=1, sign=m[i][j])
    return g


@pytest.fixture
def lowrank_multigraph(request):
    return to_multigraph(lowrank_graph(request))


@pytest.fixture
def status_0(request):
    g = lowrank_graph(request)
    s = Status()
    s.init(to_multigraph(g))
    return s


@pytest.fixture
def A6():
    A = np.zeros((6, 6))

    for i in [2, 3]:
        A[0, i] = A[i, 0] = A[1, i] = A[i, 1] = -1

    for i in [4]:
        A[0, i] = A[i, 0] = A[1, i] = A[i, 1] = 1

    for i in [5]:  # not a triangle, ignore
        A[0, i] = A[i, 0] = 1
    
    return sparse.csr_matrix(A)


@pytest.fixture
def g6():
    return matrix2graph(A6(), None, multigraph=False)



## correlation clustering fixtures


@pytest.fixture
def cc_g1():
    g = nx.Graph()
    g.add_edges_from([
        (0, 1, {'sign': 1}),
        (2, 3, {'sign': 1}),
        (4, 5, {'sign': 1}),
        (1, 2, {'sign': -1}),
        (3, 4, {'sign': -1}),
    ])
    return g


@pytest.fixture
def cc_g2():
    g = nx.Graph()
    g.add_edges_from([
        (0, 1, {'sign': 1}),
        (0, 2, {'sign': 1}),
        (1, 2, {'sign': -1}),
    ])
    return g


@pytest.fixture
def cc_g3():
    g = nx.Graph()
    g.add_edges_from([
        (0, 1, {'sign': 1}),
        (0, 2, {'sign': 1}),
        (1, 2, {'sign': -1}),
        (1, 3, {'sign': 1}),
        (2, 3, {'sign': 1}),
    ])
    return g


@pytest.fixture
def scc_g1():
    g = cc_g1()
    g.add_edges_from([
        (3, 6, {'sign': 1}),
        (4, 7, {'sign': 1}),
        (0, 8, {'sign': -1})
    ])
    return g


@pytest.fixture
def scc_g3():
    g = cc_g3()
    g.add_edges_from([
        (1, 4, {'sign': 1}),
        (4, 8, {'sign': 1}),
        (8, 9, {'sign': 1}),
        (3, 5, {'sign': 1}),
        (2, 6, {'sign': -1}),
        (0, 7, {'sign': -1})
    ])
    return g
