import contexts as ctx
import pytest
import numpy as np

from contexts import spark_context
from numpy.testing import assert_almost_equal
from snpp.cores.lowrank import alq, alq_spark
from snpp.utils.matrix import indexed_entries

from data import Q1, sparse_Q1, Q1_result


def test_lowrank_alq_dense(Q1, Q1_result):
    for m in ["random", "svd"]:
        X, Y, _ = alq(Q1, k=2, lambda_=0.1,
                      max_iter=20,
                      init_method=m)
        
        assert_almost_equal(np.sign(np.dot(X, Y)),
                            Q1_result)


pytestmark = pytest.mark.usefixtures("spark_context")

def test_lowrank_alq_spark(Q1, sparse_Q1, Q1_result, spark_context):
    """
    Borrowed from here:
    https://github.com/kawadia/pyspark.test/blob/master/examples/wordcount_test.py
    """
    edges = indexed_entries(sparse_Q1)
    edges_rdd = spark_context.parallelize(edges)
    X, Y = alq_spark(edges_rdd, rank=2, lambda_=0.1, iterations=20)
    assert_almost_equal(np.sign(np.dot(X, Y)),
                        Q1_result)
