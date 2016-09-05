import networkx as nx
import numpy as np

from tqdm import tqdm
from scipy.sparse import isspmatrix_csr
from collections import Counter


def extract_nodes_and_signs(e, e1, e2):
    """
    e: the edge in question
    e1, e2: (ni, nj, s)

    order does not matter
    """
    # nodes = {e[0], e[1], e1[0], e1[1], e2[0], e2[1]}
    # assert len(nodes) == 3
    # signs = [i[2] for i in (e, e1, e2) if len(i) > 2]
    return (set(e1[:2] + e2[:2] + e), [e1[-1], e2[-1]])


def in_different_partitions(nodes, C):
    return len(set(C[n] for n in nodes)) == len(nodes)


NEG11 = [-1, -1]


def get_sign_1st_order(e, e1, e2, C):
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    if len(set(C[n] for n in nodes)) == 3 and signs == NEG11:
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
    
    A: sign matrix (sparse,csr or lil)
    C: cluster label array
    T: target edges

    Returns:
    generator of (n_i, n_j, sign, count)
        note that (n_1, n_j) \in T
    """
    assert isspmatrix_csr(A)
    A_lil = A.tolil()  # fast single element indexing
    print("greedy -> first_order_triangles_count:")

    # caching `nonzero`
    nonzero_d = {}
    nodes = set([i for e in T for i in e])
    for n in nodes:
        nonzero_d[n] = set(A[n, :].nonzero()[1])

    for ni, nj in T:
        idx1 = nonzero_d[ni]
        idx2 = nonzero_d[nj]

        nks = filter(lambda nk: A_lil[ni, nk] != 0 and A_lil[nj, nk] != 0,
                     idx1.intersection(idx2) - {ni, nj})
        counter_by_sign = Counter()
        e = (ni, nj)
        for nk in nks:
            e1 = (ni, nk, A_lil[ni, nk])
            e2 = (nj, nk, A_lil[nj, nk])
            correct_sign = get_sign_1st_order(e, e1, e2, C)
            counter_by_sign[correct_sign] += 1
        for sign, count in counter_by_sign.items():
            yield (ni, nj, sign, count)


def first_order_triangles_count_g(g, C, T):
    """
    Args:
    
    g: nx.Graph
    C: cluster label array
    T: target edges

    Returns:
    generator of (n_i, n_j, sign, count)
        note that (n_1, n_j) \in T
    """
    assert isinstance(g, nx.Graph)

    for ni, nj in tqdm(T):
        counter_by_sign = Counter()
        e = (ni, nj)
        nks = set(g.adj[ni]).intersection(set(g.adj[nj])) - {ni, nj}
        for nk in nks:
            e1 = (ni, nk, g[ni][nk]['sign'])
            e2 = (nj, nk, g[nj][nk]['sign'])
            correct_sign = get_sign_1st_order(e, e1, e2, C)
            counter_by_sign[correct_sign] += 1
        for sign, count in counter_by_sign.items():
            yield (ni, nj, sign, count)


def first_order_triangles_net_count_g(g, C, T):
    """
    net count: #balanced - #imbalanced

    Args:
    
    g: nx.Graph
    C: cluster label array
    T: target edges

    Returns:
    generator of (n_i, n_j, sign, net_count, count_by_sign)
        note that (n_1, n_j) \in T
    """
    assert isinstance(g, nx.Graph)

    for ni, nj in tqdm(T):
        count_by_sign = np.zeros(2, dtype=np.int)  # pos 0 for -1, pos 1 for 1
        e = (ni, nj)
        nbi = set(g.adj[ni])
        nbj = set(g.adj[nj])
        nks = nbi.intersection(nbj) - {ni, nj}
        for nk in nks:
            e1 = (ni, nk, g[ni][nk]['sign'])
            e2 = (nj, nk, g[nj][nk]['sign'])
            correct_sign = get_sign_1st_order(e, e1, e2, C)
            if correct_sign == -1:
                pos = 0
            else:
                pos = 1
            count_by_sign[pos] += 1
            
        ind = np.argmax(count_by_sign)
        net_count = abs(count_by_sign[0] - count_by_sign[1])
        if ind == 0:
            sign = -1
        else:
            sign = 1
        yield (ni, nj, sign, net_count, tuple(count_by_sign))

            
