import contexts as ctx

import pytest
import random
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.metrics import adjusted_rand_score

from numpy.testing import assert_almost_equal
from snpp.cores.lowrank import weighted_partition_sparse
from snpp.cores.budget_allocation import exponential_budget
from snpp.cores.max_balance import greedy
from snpp.utils.matrix import zero, difference_ratio
from contexts import spark_context

from snpp.cores.joint_part_pred import iterative_approach, \
    single_run_approach

from data import rand_lowrank_mat, true_lowrank_mat


N = 12
rank = 3
known_edge_percentage = 0.3
random_seed = 1234
true_labels = [g for g in range(rank) for i in range(int(N / rank))]

def parameters(rand_lowrank_mat):
    A = csr_matrix(rand_lowrank_mat)
    W = np.ones(A.shape)
    T = set(zip(*zero(A)))
    k = 3
    return A, W, T, k


def test_iterative_approach(rand_lowrank_mat,
                            true_lowrank_mat,
                            spark_context):
    A, W, T, k = parameters(rand_lowrank_mat)
    C, P = iterative_approach(
        A, W, T, k,
        graph_partition_f=weighted_partition_sparse,
        graph_partition_kwargs=dict(sc=spark_context, lambda_=0.1, iterations=20,
                                    seed=random_seed),
        budget_allocation_f=exponential_budget,
        budget_allocation_kwargs=dict(exp_const=2),
        solve_maxbalance_f=greedy)
    assert isspmatrix_csr(P)

    pred_mat = (A + P).toarray()
    print(pred_mat)
    error_rate = difference_ratio(pred_mat, true_lowrank_mat)
    print(error_rate)
    
    assert error_rate < 0.05

    # arc = adjusted_rand_score(true_labels, C)
    # print(C)
    # print(arc)
    # assert arc == 1.0

    
def test_single_run_approach(rand_lowrank_mat,
                             true_lowrank_mat,
                             spark_context):
    A, W, T, k = parameters(rand_lowrank_mat)
    
    C, P = single_run_approach(A, W, T, k,
                               graph_partition_f=weighted_partition_sparse,
                               graph_partition_kwargs=dict(sc=spark_context, lambda_=0.1, iterations=20,
                                                           seed=random_seed))
    assert isspmatrix_csr(P)
    print((A + P).toarray())

    error_rate = difference_ratio((A + P).toarray(), true_lowrank_mat)
    print(error_rate)
    assert error_rate > 0.15  # 0.16666666666666666

    # ars = adjusted_rand_score(true_labels, C)
    # assert ars < 0.6  # 0.5119453924914675
