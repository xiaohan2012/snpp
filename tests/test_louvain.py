import contexts as ctx

import pytest
import numpy as np
import networkx as nx

from numpy.testing import assert_allclose
from itertools import repeat, chain
from scipy.sparse import csr_matrix

from snpp.cores.louvain import best_partition, Status, \
    __modularity, modularity, \
    __remove, __insert, __neighcom, \
    induced_graph, \
    __one_level, \
    split_graph_by_sign

from snpp.utils.data import make_lowrank_matrix
from snpp.utils.signed_graph import to_multigraph

group_size = 10
rank = 4
N = group_size * rank


def test_detect_community(lowrank_graph):
    expected = np.array(list(chain(*(repeat(i, group_size) for i in range(rank)))))  # 0,0..1,1..2,2..3,3..

    part = best_partition(lowrank_graph, 4)
    coms = [part[i] for i in range(N)]
    print(part)
    assert_allclose(coms, expected)


group_size = 2
rank = 2
N = group_size * rank


def test_status_0(status_0):
    assert status_0.degrees_p == {i: 3 for i in range(N)}
    assert status_0.degrees_n == {i: 2 for i in range(N)}
    assert status_0.gdegrees_p == {i: 3 for i in range(N)}
    assert status_0.gdegrees_n == {i: 2 for i in range(N)}
    assert status_0.loops_p == {i: 1 for i in range(N)}  # $\sum_{in}$
    assert status_0.loops_n == {i: 0 for i in range(N)}
    assert status_0.internals_p == {i: 1 for i in range(N)}  # $\sum_{in}$
    assert status_0.internals_n == {i: 0 for i in range(N)}
    assert status_0.total_weight_p == 6
    assert status_0.total_weight_n == 4


def test_status_true(lowrank_multigraph):
    """correct partition
    """
    
    s = Status()
    part = {0: 0, 1: 0, 2: 1, 3: 1}
    s.init(lowrank_multigraph, part)

    assert s.degrees_p == {0: 6, 1: 6}
    assert s.degrees_n == {0: 4, 1: 4}
    assert s.gdegrees_p == {i: 3 for i in range(N)}
    assert s.gdegrees_n == {i: 2 for i in range(N)}
    assert s.loops_p == {i: 1 for i in range(N)}
    assert s.loops_n == {i: 0 for i in range(N)}
    assert s.internals_p == {0: 3, 1: 3}
    assert s.internals_n == {0: 0, 1: 0}
    assert s.total_weight_p == 6
    assert s.total_weight_n == 4
    assert s.node2com == part


def test_status_false(lowrank_multigraph):
    """incorrect partition
    """
    s = Status()
    part = {0: 0, 1: 1, 2: 0, 3: 1}
    s.init(lowrank_multigraph, part)

    assert s.degrees_p == {0: 6, 1: 6}
    assert s.degrees_n == {0: 4, 1: 4}
    assert s.gdegrees_p == {i: 3 for i in range(N)}
    assert s.gdegrees_n == {i: 2 for i in range(N)}
    assert s.loops_p == {i: 1 for i in range(N)}
    assert s.loops_n == {i: 0 for i in range(N)}
    assert s.internals_p == {0: 2, 1: 2}
    assert s.internals_n == {0: 1, 1: 1}
    assert s.total_weight_p == 6
    assert s.total_weight_n == 4
    assert s.node2com == part

    
def test_modularity(lowrank_multigraph, status_0):
    # vanila calculation
    Q_p = 0
    Q_n = 0
    m_p, m_n =  status_0.total_weight_p, status_0.total_weight_n
    for i in range(N):
        for j in range(N):
            if i == j:
                Q_p += (1 - (status_0.degrees_p[i]**2 / (2*m_p)))
                Q_n += (0 - (status_0.degrees_n[i]**2 / (2*m_n)))
    Q_p /= (2*m_p)
    Q_n /= (2*m_n)
    
    assert (Q_p - Q_n) == __modularity(status_0)
    assert modularity({i: i for i in range(N)}, lowrank_multigraph) == (Q_p - Q_n)

    
def test_modularity_given_partition(lowrank_multigraph):
    s = Status()
    part = {0: 0, 1: 0, 2: 1, 3: 1}
    s.init(lowrank_multigraph, part)

    assert __modularity(s) == modularity(part, lowrank_multigraph)


def test_neighcom(lowrank_multigraph):
    s = Status()
    part = {0: 0, 1: 1, 2: 1, 3: 2}
    s.init(lowrank_multigraph, part)
    d = __neighcom(0, lowrank_multigraph, s)
    assert d == {1: [1, 1], 2: [0, 1]}

    s = Status()
    part = {0: 0, 1: 1, 2: 1, 3: 1}
    s.init(lowrank_multigraph, part)
    d = __neighcom(0, lowrank_multigraph, s)
    assert d == {1: [1, 2]}


def test_remove(lowrank_multigraph):
    part = {0: 0, 1: 1, 2: 1, 3: 1}
    s = Status()
    s.init(lowrank_multigraph, part)
    __remove(node=1, com=1, weight_p=0, weight_n=2, status=s)

    s1 = Status()
    s1.init(lowrank_multigraph, {0: 0, 1: -1, 2: 1, 3: 1})
    del s1.degrees_p[-1]
    del s1.degrees_n[-1]
    del s1.internals_p[-1]
    del s1.internals_n[-1]
    assert s.degrees_p == s1.degrees_p
    assert s.degrees_n == s1.degrees_n
    assert s.internals_p == s1.internals_p
    assert s.internals_n == s1.internals_n


def test_insert(lowrank_multigraph):
    part = {0: 0, 1: -1, 2: 1, 3: 1}
    s = Status()
    s.init(lowrank_multigraph, part)
    __insert(node=1, com=0, weight_p=1, weight_n=0, status=s)

    del s.degrees_p[-1]
    del s.degrees_n[-1]
    del s.internals_p[-1]
    del s.internals_n[-1]
    
    s1 = Status()
    s1.init(lowrank_multigraph, {0: 0, 1: 0, 2: 1, 3: 1})

    assert s.degrees_p == s1.degrees_p
    assert s.degrees_n == s1.degrees_n
    assert s.internals_p == s1.internals_p
    assert s.internals_n == s1.internals_n


def test_induced_graph(lowrank_multigraph):
    part = {0: 0, 1: 0, 2: 1, 3: 1}
    g_new = induced_graph(part, lowrank_multigraph)
    goal = nx.MultiGraph()
    goal.add_edge(0, 0, key=1, weight=3, sign=1)
    goal.add_edge(1, 1, key=1, weight=3, sign=1)
    goal.add_edge(0, 1, key=-1, weight=4, sign=-1)

    equal = (lambda x, y: x == y)
    assert nx.is_isomorphic(g_new, goal,
                            node_match=equal,
                            edge_match=equal)


def test_one_level(lowrank_multigraph):
    s = Status()
    s.init(lowrank_multigraph)
    __one_level(lowrank_multigraph, s)

    part = s.node2com
    assert part[0] == part[1]
    assert part[2] == part[3]
    assert part[1] != part[3]


def test_split_graph_by_sign(lowrank_multigraph):
    gp, gn = split_graph_by_sign(lowrank_multigraph)
    assert set([(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]) == set(gp.edges())
    assert set([(0, 2), (1, 2), (0, 3), (1, 3)]) == set(gn.edges())
