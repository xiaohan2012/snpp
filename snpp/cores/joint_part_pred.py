# Approximation algorithms for joint partition and prediction (JointPartPred)

import numpy as np
from snpp.utils import nonzero_edges, predict_signs_using_partition


def iterative_approach(A, W, T, k,
                       graph_partition_f,
                       budget_allocation_f,
                       solve_maxbalance_f,
                       graph_partition_kwargs={},
                       solve_maxbalance_kwargs={}):
    """
    Params:
    
    A: edge sign matrix
    W: edge weight matrix
    T: target edge set (set of edges, (i, j))
    k: partition number

    Returns:
    
    C: partition, partition label array, 1xn
    P: predicted sign matrix on T
    """
    assert A.shape == W.shape, 'shape mismatch'
    n1, n2 = A.shape
    assert n1 == n2, 'dimension mismatch'

    iter_n = 0
    P = np.zeros((n1, n2))
    while T != nonzero_edges(P):
        iter_n += 1
        C = graph_partition_f(A + P, W, k,
                              **graph_partition_kwargs)
        B = budget_allocation_f(C, A, P, iter_n)
        P_prime = solve_maxbalance_f(A+P, W, C, B,
                                     **solve_maxbalance_kwargs)
        P += P_prime
    C = graph_partition_f(A + P, W, k, **graph_partition_kwargs)
    return C, P


def naive_approach(A, W, T, k,
                   graph_partition_f,
                   graph_partition_kwargs={}):
    C = graph_partition_f(A, W, k,
                          **graph_partition_kwargs)
    P = predict_signs_using_partition(C, targets=T)
    return C, P
    
    
