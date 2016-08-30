# Approximation algorithms for joint partition and prediction (JointPartPred)

import numpy as np

from copy import copy
from scipy.sparse import csr_matrix
from snpp.utils import nonzero_edges, predict_signs_using_partition


def iterative_approach(A, W, T, k,
                       graph_partition_f,
                       budget_allocation_f,
                       solve_maxbalance_f,
                       graph_partition_kwargs={},
                       budget_allocation_kwargs={},
                       solve_maxbalance_kwargs={}):
    """
    Params:
    
    A: edge sign matrix
    W: edge weight matrix
    T: target edge set (set of edges, (i, j))
    k: partition number

    graph_partition_f: method for graph partitioning
    budget_allocation_f: budget allocation method
    solve_maxbalance_f: method for approximating the max balance problem

    Returns:
    
    C: partition, partition label array, 1xn
    P: predicted sign matrix on T (csr_matrix)
    """
    assert A.shape == W.shape, 'shape mismatch'
    n1, n2 = A.shape
    assert n1 == n2, 'dimension mismatch'

    remaining_targets = copy(T)
    
    iter_n = 0
    P = csr_matrix((n1, n2))
    while T != nonzero_edges(P):
        iter_n += 1
        print('iteration={}, #remaining targets={}'.format(
            iter_n, len(remaining_targets)))
        C = graph_partition_f(A + P, W, k,
                              **graph_partition_kwargs)
        B = budget_allocation_f(C, A, P, iter_n, **budget_allocation_kwargs)
        P_prime = solve_maxbalance_f(A + P, W, C, B, T=remaining_targets,
                                     **solve_maxbalance_kwargs)
        remaining_targets -= set(zip(*P_prime.nonzero()))
        P += P_prime
    print("inside joint_part_pred:")
    print((A + P).toarray())
    print(k)
    C = graph_partition_f(A + P, W, k,
                          **graph_partition_kwargs)
    return C, P


def single_run_approach(A, W, T, k,
                        graph_partition_f,
                        graph_partition_kwargs={}):
    """
    """
    C = graph_partition_f(A, W, k,
                          **graph_partition_kwargs)
    P = predict_signs_using_partition(C, targets=T)
    return C, P.tocsr()
    
    
