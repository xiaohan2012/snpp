import numpy as np
from scipy.sparse import csr_matrix, issparse, dok_matrix
from sklearn.cross_validation import train_test_split


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


def difference_ratio_sparse(M1, M2):
    """different ratio on nonzero elements
    """
    assert issparse(M1)
    assert issparse(M2)
    assert M1.shape == M2.shape
    assert M1.nnz == M2.nnz

    s1 = set(indexed_entries(M1))
    s2 = set(indexed_entries(M2))
    return 1 - len(s1.intersection(s2)) / M1.nnz


def save_sparse_csr(filename, array):
    np.savez(filename,
             data=array.data,
             indices=array.indices,
             indptr=array.indptr,
             shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'],
                       loader['indices'],
                       loader['indptr']),
                      shape=loader['shape'])


def _make_matrix(items, shape):
    idx1, idx2, data = zip(*items)
    return csr_matrix((data, (idx1, idx2)), shape=shape)

    
def split_train_dev_test(m, weights=[0.8, 0.1, 0.1]):
    """
    Returns:

    three matrices whose nnz satistify weights
    """
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 0.00001
    assert issparse(m)

    entries = indexed_entries(m)

    remain_sum = np.sum(weights[1:])

    train, dev_test = train_test_split(entries, train_size=weights[0], test_size=remain_sum)

    dev, test = train_test_split(dev_test,
                                 train_size=weights[1] / remain_sum,
                                 test_size=weights[2] / remain_sum)
    
    return (_make_matrix(train, m.shape),
            _make_matrix(dev, m.shape),
            _make_matrix(test, m.shape))


def split_train_test(m, weights=[0.9, 0.1]):
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 0.00001
    assert issparse(m)

    entries = indexed_entries(m)

    train, test = train_test_split(entries, train_size=weights[0], test_size=weights[1])
    return (_make_matrix(train, m.shape),
            _make_matrix(test, m.shape))
