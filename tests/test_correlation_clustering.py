import contexts as ctx

import pytest
import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
from snpp.cores.correlation_clustering import agglomerative, \
    sampling_wrapper
from sklearn.metrics import adjusted_rand_score

kwargs = dict(return_dict=False,
              change_sign=True)


def test_agglomerative_1(cc_g1):
    C = agglomerative(cc_g1, **kwargs)
    assert_allclose(C, np.array([0, 0, 1, 1, 2, 2]))


def test_agglomerative_all_singletons(cc_g1):
    C = agglomerative(cc_g1, threshold=-1.0, **kwargs)
    assert_allclose(C, np.arange(cc_g1.number_of_nodes()))


def test_agglomerative_2(cc_g2):
    C = agglomerative(cc_g2, **kwargs)
    assert_allclose(C, [0, 0, 1])


def test_agglomerative_3(cc_g3):
    C = agglomerative(cc_g3, **kwargs)
    assert_allclose(C, np.zeros(4))


def test_agglomerative_4():
    # just one node
    g = nx.DiGraph()
    g.add_node(0)
    C = agglomerative(g, **kwargs)
    assert_allclose(C, np.zeros(1))


def test_agglomerative_5():
    # all singletons
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    C = agglomerative(g, **kwargs)
    assert_allclose(C, np.arange(4))


def test_agglomerative_6(cc_g3):
    # two singletons
    cc_g3.add_nodes_from([4, 5])
    C = agglomerative(cc_g3, **kwargs)
    assert_allclose(C, np.array([0, 0, 0, 0, 1, 2]))
    

def test_agglomerative_7():
    # empty graph
    g = nx.Graph()
    with pytest.raises(ValueError, message='empty graph'):
        agglomerative(g, **kwargs)


def test_sampling_wrapper_1(scc_g1):
    C = sampling_wrapper(scc_g1, agglomerative, samples=list(range(6)),
                         return_dict=False)
    # assert adjusted_rand_score(C, np.array([0]*8 + [1]))
    assert_allclose(C, np.array([0, 0, 1, 1, 2, 2, 1, 2, 3]))
    # raise


def test_sampling_wrapper_2(scc_g3):
    C = sampling_wrapper(scc_g3, agglomerative, samples=list(range(4)),
                         return_dict=False)
    # assert adjusted_rand_score(C, np.array([0]*8 + [1]))
    assert_allclose(C, np.array([0]*6 + [1, 2, 0, 0]))


def test_sampling_wrapper_3(scc_g3):
    C = sampling_wrapper(scc_g3, agglomerative, samples=[1, 2],
                         return_dict=False)
    assert_allclose(C, np.array([0, 0, 1] + [0]*3 + [2, 3, 0, 0]))


def test_sampling_wrapper_4(scc_g3):
    C = sampling_wrapper(scc_g3, agglomerative, samples=[1, 2],
                         return_dict=False)
    assert_allclose(C, np.array([0, 0, 1] + [0]*3 + [2, 3, 0, 0]))


def test_sampling_wrapper_bug_1():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1, sign=1)
    C = sampling_wrapper(g, agglomerative, samples=[0, 2],
                         return_dict=False)
    assert_allclose(C, [0, 0, 1])
