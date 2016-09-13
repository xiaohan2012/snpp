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
    assert adjusted_rand_score(C, np.array([0, 0, 1, 1, 2, 2])) == 1.0


def test_agglomerative_2(cc_g2):
    C = agglomerative(cc_g2, **kwargs)
    assert adjusted_rand_score(C, np.zeros(3)) == 1.0


def test_agglomerative_3(cc_g3):
    C = agglomerative(cc_g3, **kwargs)
    assert adjusted_rand_score(C, np.zeros(4)) == 1.0


def test_agglomerative_4():
    # just one node
    g = nx.DiGraph()
    g.add_node(0)
    C = agglomerative(g, **kwargs)
    assert adjusted_rand_score(C, np.zeros(1)) == 1.0


def test_agglomerative_5():
    # all singletons
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    C = agglomerative(g, **kwargs)
    assert adjusted_rand_score(C, np.arange(4)) == 1.0


def test_agglomerative_6(cc_g3):
    # two singletons
    cc_g3.add_nodes_from([4, 5])
    C = agglomerative(cc_g3, **kwargs)
    assert adjusted_rand_score(C, np.array([0, 0, 0, 0, 1, 2])) == 1.0
    

def test_agglomerative_7():
    # empty graph
    g = nx.Graph()
    with pytest.raises(ValueError, message='empty graph'):
        agglomerative(g, **kwargs)


def test_sampling_wrapper_1(scc_g1):
    C = sampling_wrapper(scc_g1, agglomerative, samples=list(range(6)),
                         return_dict=False)
    assert adjusted_rand_score(C, np.array([0]*8 + [1]))
    # raise


def test_sampling_wrapper_2(scc_g3):
    C = sampling_wrapper(scc_g3, agglomerative, samples=list(range(4)),
                         return_dict=False)
    assert adjusted_rand_score(C, np.array([0]*6 + [1, 2]))
    
