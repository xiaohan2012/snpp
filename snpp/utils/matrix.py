import numpy as np


def indexed_entries(sparse_matrix):
    """
    Args:
    
    Return:
    list of (row_id, col_id, value)
    """
    return [(i, j, sparse_matrix[i, j])
            for i, j in zip(*sparse_matrix.nonzero())]


def zero(m):
    """return the zero-valued entries' indices
    """
    return (m == 0).nonzero()


def difference_ratio(M1, M2):
    assert M1.shape == M2.shape
    _, idx = np.nonzero(M1 != M2)
    return len(idx) / M1.size
