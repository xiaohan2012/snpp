import numpy as np
from collections import Counter


def extract_nodes_and_signs(e, e1, e2):
    """
    e: the edge in question
    e1, e2: (ni, nj, s)
    """    
    nodes = {e[0], e[1], e1[0], e1[1], e2[0], e2[1]}
    assert len(nodes) == 3
    signs = [i[2] for i in (e, e1, e2) if len(i) > 2]
    return nodes, signs


def in_different_partitions(nodes, C):
    return len(set(C[n] for n in nodes)) == len(nodes)


# Deprecated
def can_be_balanced(e, e1, e2, C):
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    if in_different_partitions(nodes, C):
        return signs == [-1, -1]
    return True


def get_sign_1st_order(e, e1, e2, C):
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    
    if in_different_partitions(nodes, C) and signs == [-1, -1]:
        # weak balance
        return -1
    else:
        # strong balance
        neg_cnt = len(list(filter(lambda s: s == -1, signs)))
        if neg_cnt % 2 == 0:
            return 1
        else:
            return -1


def first_order_triangles_count(A, C, T):
    """
    Args:
    
    A: sign matrix (sparse)
    C: cluster label array
    T: target edges

    Returns:
    generator of (n_i, n_j, sign, count)
        note that (n_1, n_j) \in T
    """
    for ni, nj in T:
        # seems no support for bit-wise operation on scipy.sparse
        adj_vect = np.logical_and(A[ni, :].todense(),
                                  A[nj, :].todense())
        adj_vect[0, ni] = adj_vect[0, nj] = 0
        _, nks = adj_vect.nonzero()
        
        counter_by_sign = Counter()  # cleared after each main loop
        for nk in nks:
            if A[ni, nk] != 0 and A[nj, nk] != 0:  # edges not missing
                e = (ni, nj)
                e1 = (ni, nk, A[ni, nk])
                e2 = (nj, nk, A[nj, nk])
                correct_sign = get_sign_1st_order(e, e1, e2, C)
                counter_by_sign[correct_sign] += 1
        for sign, count in counter_by_sign.items():
            yield (ni, nj, sign, count)
