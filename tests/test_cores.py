import contexts as ctx
import numpy as np

from numpy.testing import assert_almost_equal
from snpp.cores.lowrank import alq, alq_sparse

from data import Q1, sparse_Q1, Q1_result


def test_lowrank_alq_dense(Q1, Q1_result):
    for m in ["random", "svd"]:
        X, Y, _ = alq(Q1, k=2, lambda_=0.1,
                      max_iter=20,
                      init_method=m)
        
        assert_almost_equal(np.sign(np.dot(X, Y)),
                            Q1_result)


# def test_lowrank_alq_sparse(Q1, sparse_Q1, Q1_result):
#     for m in ["random", "svd"]:
#         ctx.reset_random_seed()
#         X, Y, _ = alq_sparse(sparse_Q1, k=2, lambda_=0.1, max_iter=10,
#                              init_method=m)
        
#         ctx.reset_random_seed()
#         X_true, Y_true, _ = alq(Q1, k=2, lambda_=0.1, max_iter=10,
#                                 init_method=m)

#         assert_almost_equal(X_true.dot(Y_true), np.dot(X, Y))

#         assert_almost_equal(np.sign(np.dot(X, Y)),
#                             Q1_result)
