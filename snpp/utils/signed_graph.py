import networkx as nx
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix
from snpp.utils.matrix import indexed_entries


def make_symmetric(m):
    """for entries whose diagonal counterparts are missing,
    replicate the values

    Return csr_matrix
    """
    entries_to_add = []
    for i, j in zip(*m.nonzero()):
        if m[j, i] == 0:
            entries_to_add.append((j, i, m[i, j]))
    idx1, idx2, data = zip(*(indexed_entries(m) + entries_to_add))
    return csr_matrix((data, (idx1, idx2)), shape=m.shape)


def fill_diagonal(m, val=1):
    assert issparse(m)
    assert m.shape[0] == m.shape[1]
    m_new = m.todok()

    for i in range(m_new.shape[0]):
        m_new[i, i] = val
    return m_new.tocsr()


def symmetric_stat(m):
    """
    1. number of edges that has a symmetric one (regardless of the sign)
    2. number of edges that have the same sign with its symmetric counterpart
    """
    assert issparse(m)
    c1 = 0
    c2 = 0
    for i, j in zip(*m.nonzero()):
        if m[j, i] != 0 and i != j:
            c1 += 1
            if m[i, j] == m[j, i]:
                c2 += 1
    return c1, c2


def matrix2graph(A, W):
    """returns MultiGraph
    """
    print('building MultiGraph')
    g = nx.MultiGraph()  # allow parallel edges
    idxs = tqdm(zip(*A.nonzero()))
    if W is None:
        for i, j in idxs:
            g.add_edge(i, j, key=A[i, j], weight=1, sign=A[i, j])
    else:
        for i, j in idxs:
            g.add_edge(i, j, key=A[i, j], weight=W[i, j], sign=A[i, j])
    return g
    
