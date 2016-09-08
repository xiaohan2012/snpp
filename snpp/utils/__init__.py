import numpy as np
from itertools import combinations


def nonzero_edges(A):
    return set(zip(*np.nonzero(A)))


def predict_signs_using_partition(C, targets=None):
    """
    Using weak-balanced theory to predict edge signs
    Params:
    
    C: the partition label array
    targets: the target edge set
    
    Returns:

    list of (i, j, sign)
    """

    preds = []
    idx = list(range(len(C)))

    if targets is None:
        targets = combinations(idx, 2)
    targets = set(targets)
    
    for i, j in targets:
        if C[i] == C[j]:
            s = 1
        else:
            s = -1
        preds.append((i, j, s))
    return preds














