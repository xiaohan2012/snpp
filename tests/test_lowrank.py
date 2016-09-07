import contexts as ctx

import pytest
import numpy as np
import networkx as nx

from numpy.testing import assert_almost_equal, assert_allclose
from sklearn.metrics import adjusted_rand_score

from snpp.cores.lowrank import alq, alq_spark, \
    alq_weighted_spark, \
    partition_graph, \
    partition_sparse


random_seed = 123456


def test_lowrank_alq_dense(Q1, Q1_result):
    # assert_allclose(Q1, np.transpose(Q1))
    for m in ["random", "svd"]:
        X, Y, _ = alq(Q1, k=2, lambda_=0.1,
                      max_iter=30,
                      init_method=m)
        # assert_allclose(X, np.transpose(Y), rtol=0.3)
        assert_almost_equal(np.sign(np.dot(X, Y)),
                            Q1_result)


def test_lowrank_alq_spark(sparse_Q1, Q1_result, spark_context):
    """
    Borrowed from here:
    https://github.com/kawadia/pyspark.test/blob/master/examples/wordcount_test.py
    """
    X, Y = alq_spark(sparse_Q1, k=2, sc=spark_context,
                     lambda_=0.1, iterations=20,
                     seed=random_seed)
    assert_almost_equal(np.sign(np.dot(X, Y)),
                        Q1_result)

    X, Y = alq_weighted_spark(sparse_Q1, None, 2, spark_context,
                              lambda_=0.1, iterations=20)
    assert_almost_equal(np.sign(np.dot(X, Y)),
                        Q1_result)


def test_partition_graph_simple(g1, spark_context):
    labels = partition_graph(g1, k=2,
                             sc=spark_context,
                             iterations=20, lambda_=0.1,
                             seed=random_seed)
    assert adjusted_rand_score(labels, [0, 0, 1, 1]) == 1.0

N = 12
rank = 3
known_edge_percentage = 0.5  # should be lower, isn't
true_labels = [g for g in range(rank) for i in range(int(N / rank))]


def test_partition_graph(rand_lowrank_g, spark_context):
    print(nx.to_scipy_sparse_matrix(rand_lowrank_g, weight='sign').todense())
    labels = partition_graph(rand_lowrank_g,
                             k=rank,
                             sc=spark_context,
                             iterations=20, lambda_=0.1,
                             seed=random_seed)
    assert adjusted_rand_score(labels, true_labels) == 1.0


def test_partition_graph_1(rand_lowrank_g, rand_lowrank_mat,
                           spark_context):
    """compared with original version
    """
    labels_new = partition_graph(rand_lowrank_g,
                                 k=rank,
                                 sc=spark_context,
                                 iterations=20, lambda_=0.1,
                                 seed=random_seed)
    labels_old = partition_sparse(rand_lowrank_mat,
                                  k=rank,
                                  sc=spark_context,
                                  iterations=20, lambda_=0.1,
                                  seed=random_seed)
    assert adjusted_rand_score(labels_new, labels_old) == 1.0
