import contexts as ctx
import pytest
import numpy as np

from contexts import spark_context
from numpy.testing import assert_almost_equal
from sklearn.metrics import adjusted_rand_score

from snpp.cores.lowrank import alq, alq_spark, \
    alq_weighted_spark,\
    weighted_partition_sparse
from snpp.utils.matrix import indexed_entries

from data import Q1, sparse_Q1, Q1_result, rand_lowrank_mat


random_seed = 123456


def test_lowrank_alq_dense(Q1, Q1_result):
    for m in ["random", "svd"]:
        X, Y, _ = alq(Q1, k=2, lambda_=0.1,
                      max_iter=20,
                      init_method=m)
        
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


N = 12
rank = 3
known_edge_percentage = 0.9
true_labels = [g for g in range(rank) for i in range(int(N / rank))]


def test_weighted_partition_sparse(rand_lowrank_mat, spark_context):
    print(rand_lowrank_mat)
    labels = weighted_partition_sparse(rand_lowrank_mat, None, k=rank,
                                       sc=spark_context,
                                       iterations=20, lambda_=0.1,
                                       seed=random_seed)
    assert adjusted_rand_score(labels, true_labels) == 1.0

