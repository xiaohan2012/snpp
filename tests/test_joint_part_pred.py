import contexts as ctx

import pytest
import random
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.metrics import adjusted_rand_score

from numpy.testing import assert_almost_equal, assert_allclose
from snpp.cores.lowrank import partition_graph
from snpp.cores.budget_allocation import exponential_budget
from snpp.cores.max_balance import greedy_g
from snpp.cores.louvain import best_partition
from snpp.utils.matrix import zero, difference_ratio
from contexts import spark_context

from snpp.cores.joint_part_pred import iterative_approach, \
    single_run_approach

from data import rand_lowrank_mat, true_lowrank_mat, \
    rand_lowrank_g, true_lowrank_g


N = 12
rank = 3
known_edge_percentage = 0.4
random_seed = 1234
true_labels = [g for g in range(rank) for i in range(int(N / rank))]


def parameters(rand_lowrank_g):
    g = rand_lowrank_g
    zeros = set(zip(*zero(nx.to_scipy_sparse_matrix(g, weight='sign'))))
    T = set()
    for i, j in zeros:
        e = ((i, j) if i < j else (j, i))
        if e not in T:
            T.add(e)
    k = 3
    return g, T, k


def get_accuracy(true_lowrank_g, preds):
    truth = []
    for i, j, s in preds:
        truth.append((i, j, true_lowrank_g[i][j]['sign']))
    return len(set(preds).intersection(set(truth))) / len(preds)

    
def test_iterative_approach(rand_lowrank_g,
                            true_lowrank_g,
                            spark_context):
    g, T, k = parameters(rand_lowrank_g)
    for i in range(g.number_of_nodes()):
        assert g[i][i]['sign'] == 1
        
    C, preds = iterative_approach(
        g, T, k,
        graph_partition_f=partition_graph,
        graph_partition_kwargs=dict(sc=spark_context, lambda_=0.1, iterations=20,
                                    seed=random_seed),
        budget_allocation_f=exponential_budget,
        budget_allocation_kwargs=dict(exp_const=2),
        solve_maxbalance_f=greedy_g)
    assert_allclose(get_accuracy(true_lowrank_g, preds),
                    1.0)


def test_iterative_approach_louvain(rand_lowrank_g,
                                    true_lowrank_g,
                                    spark_context):
    """using louvain algorithm for graph partitioning
    """
    g, T, k = parameters(rand_lowrank_g)
    C, preds = iterative_approach(
        g, T, k,
        graph_partition_f=best_partition,
        budget_allocation_f=exponential_budget,
        budget_allocation_kwargs=dict(exp_const=2),
        solve_maxbalance_f=greedy_g)

    assert_allclose(get_accuracy(true_lowrank_g, preds),
                    0.675)

    
def test_single_run_approach(rand_lowrank_g,
                             true_lowrank_g,
                             spark_context):
    g, T, k = parameters(rand_lowrank_g)
    
    C, preds = single_run_approach(g, T, k,
                                   graph_partition_f=partition_graph,
                                   graph_partition_kwargs=dict(sc=spark_context, lambda_=0.1, iterations=20,
                                                               seed=random_seed))
    print(C)

    assert_allclose(get_accuracy(true_lowrank_g, preds),
                    1.0)
