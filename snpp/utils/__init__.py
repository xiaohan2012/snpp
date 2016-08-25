import numpy as np
from itertools import permutations


def nonzero_edges(A):
    return set(zip(*np.nonzero(A)))


def predict_signs_using_partition(C, targets=None):
    """
    Using weak-balanced theory to predict edge signs
    Params:
    
    C: the partition label array
    targets: the target edge set
    
    Returns:

    Sign matrix on targets
    """

    n = len(C)
    P = np.zeros((n, n))
    idx = list(range(len(C)))

    if targets is None:
        targets = permutations(idx, 2)
    targets = set(targets)
    
    for i, j in targets:
        if C[i] == C[j]:
            P[i, j] = 1
        else:
            P[i, j] = -1

    for i in idx:
        P[i, i] = 1
    return P














