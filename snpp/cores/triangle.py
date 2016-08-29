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


def can_be_balanced(e, e1, e2, C):
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    if in_different_partitions(nodes, C):
        return signs == [-1, -1]
    return True


def get_sign_1st_order(e, e1, e2, C):
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    
    if in_different_partitions(nodes, C):
        assert can_be_balanced(e, e1, e2, C)
        return -1
    else:
        neg_cnt = len(filter(lambda s: s == -1, signs))
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
    result: dict of (n_i, n_j, s) -> # balanced count,
       (n_1, n_j) \in T
    """
    result = Counter()
    for ni, nj in T:
        adj_vect = A[ni, :] + A[nj, :]
        adj_vect[0, ni] = adj_vect[0, nj] = 0
        _, nks = adj_vect.nonzero()
        for nk in nks:
            if A[ni, nk] != 0 and A[nj, nk] != 0:  # edges not missing
                e = (ni, nj)
                e1 = (ni, nk, A[ni, nk])
                e2 = (nj, nk, A[nj, nk])
                if can_be_balanced(e, e1, e2, C):
                    correct_sign = get_sign_1st_order(e, e1, e2, C)
                    result[(ni, nj, correct_sign)] += 1
