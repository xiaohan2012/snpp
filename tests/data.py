import random
import pytest

import networkx as nx
import numpy as np
from scipy import sparse

from snpp.utils.data import example_for_intuition, make_lowrank_matrix
from snpp.cores.louvain import Status


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
    return M


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
    return M


def make_signed_matrix(N, friends, enemies):
    Q = np.zeros((N, N))
    for i, j in friends:
        Q[i-1, j-1] = Q[j-1, i-1] = 1
    for i, j in enemies:
        Q[i-1, j-1] = Q[j-1, i-1] = -1
    for i in range(N):
        Q[i, i] = 1
        
    return Q


@pytest.fixture
def Q1():
    """a simple signed matrix
    """
    random.seed(12345)
    np.random.seed(12345)

    N = 4
    friends = [(1, 2), (3, 4)]
    enemies = [(1, 3)]
    return make_signed_matrix(N, friends, enemies)


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
def lowrank_graph(request):
    group_size = module_attr(request, 'group_size', 10)
    rank = module_attr(request, 'rank', 4)

    m = make_lowrank_matrix(group_size, rank)
    
    g = nx.MultiGraph()  # allow parallel edges
    n_row, n_col = m.shape
    for i in range(n_row):
        for j in range(n_col):
            s = m[i][j]
            g.add_edge(i, j, key=s, weight=1, sign=s)
    return g

@pytest.fixture
def status_0(request):
    g = lowrank_graph(request)
    s = Status()
    s.init(g)
    return s
