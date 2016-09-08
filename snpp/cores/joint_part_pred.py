# Approximation algorithms for joint partition and prediction (JointPartPred)

import numpy as np

from copy import copy
from scipy.sparse import csr_matrix
from snpp.utils import nonzero_edges, predict_signs_using_partition
from snpp.utils.signed_graph import matrix2graph
from snpp.utils.status import Status


def iterative_approach(g, T, k,
                       graph_partition_f,
                       budget_allocation_f,
                       solve_maxbalance_f,
                       graph_partition_kwargs={},
                       budget_allocation_kwargs={},
                       solve_maxbalance_kwargs={},
                       truth=None,
                       perform_last_partition=True):
    """
    Params:
    
    g: networkx.Graph (**mutable**)
    T: target edge set (set of edges, (i, j))
       the i, j order doesn't matter because it's undirected
    k: partition number

    graph_partition_f: method for graph partitioning
    budget_allocation_f: budget allocation method
    solve_maxbalance_f: method for approximating the max balance problem

    truth: set of (i, j, s), the ground truth for targets
        for debugging purpose

    Returns:
    
    C: partition, partition label array, 1xn
    predictions: list of (i, j, sign)
    status:
    """
    T = set(T)
    remaining_targets = copy(T)

    if truth:
        edge2true_sign = {}
        for n1, n2, v in truth:
            assert v != 0
            edge2true_sign[(n1, n2)] = v
        solve_maxbalance_kwargs['edge2true_sign'] = edge2true_sign
        status = Status()

    iter_n = 0
    all_predictions = []

    while len(remaining_targets) > 0:
        iter_n += 1
        print('iteration={}, #remaining targets={}'.format(
            iter_n, len(remaining_targets)))

        print("graph partitioning...")
        C = graph_partition_f(g, k,
                              **graph_partition_kwargs)
        
        B = budget_allocation_f(C, g, iter_n, **budget_allocation_kwargs)
        
        print("solving max_balance")
        predictions = solve_maxbalance_f(g, C, B, T=remaining_targets,
                                         **solve_maxbalance_kwargs)

        all_predictions += predictions
        remaining_targets -= set((i, j) for i, j, _ in predictions)
        g.add_edges_from((i, j, {'weight': 1, 'sign': s})
                         for i, j, s in predictions)
                
        if truth:
            acc = len(truth.intersection(set(all_predictions))) / len(all_predictions)
            print('Accuracy on {} predictions is {}'.format(
                len(all_predictions), acc
            ))
            status.update(predictions, acc, C)

    if perform_last_partition:
        C = graph_partition_f(g, k,
                              **graph_partition_kwargs)
    if truth:
        return C, all_predictions, status
    else:
        return C, all_predictions


def single_run_approach(g, T, k,
                        graph_partition_f,
                        graph_partition_kwargs={}):
    """
    """
    C = graph_partition_f(g, k,
                          **graph_partition_kwargs)
    preds = predict_signs_using_partition(C, targets=T)
    return C, preds
