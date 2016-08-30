from scipy.sparse import issparse

def make_symmetric(m):
    pass


def fill_diagonal(m, val=1):
    pass


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
