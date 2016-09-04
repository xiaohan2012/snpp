import contexts as ctx

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import isspmatrix_dok
from snpp.utils import nonzero_edges, \
    predict_signs_using_partition
from snpp.utils.data import make_lowrank_matrix
from snpp.utils.matrix import zero, \
    split_train_dev_test, \
    split_train_test, \
    difference_ratio, \
    difference_ratio_sparse, \
    delete_csr_entries
from data import random_graph, sparse_Q1, Q1_d


def test_zeros():
    xs, ys = zero(np.eye(2))
    assert xs.tolist() == [0, 1]
    assert ys.tolist() == [1, 0]


def test_nonzero_edges():
    n = 3
    M = np.eye(n)
    assert nonzero_edges(M) == {(i, i) for i in range(n)}


def test_predict_signs_using_partition():
    rank = 2  # cluster count
    size = 2  # cluster size
    C = [i for i in range(rank) for j in range(size)]

    # without targets
    P = predict_signs_using_partition(C, targets=None)
    assert isspmatrix_dok(P)

    true_P = make_lowrank_matrix(size, rank=rank) - np.eye(size * rank)
    np.testing.assert_almost_equal(P.toarray(), true_P)

    # with targets
    P = predict_signs_using_partition(C, targets=[(0, 2), (2, 0)])
    assert isspmatrix_dok(P)
    
    true_P = np.array([[0, 0, -1, 0],
                       [0, 0, 0, 0],
                       [-1, 0, 0, 0],
                       [0, 0, 0, 0]])
    
    np.testing.assert_almost_equal(P.toarray(), true_P)


def test_split_train_dev_test(random_graph):
    m = random_graph
    weights = [0.7, 0.2, 0.1]
    train, dev, test = split_train_dev_test(m, weights=weights)
    sizes = np.array([train.nnz, dev.nnz, test.nnz])

    assert_allclose(sizes / np.sum(sizes), weights, atol=0.02)
    assert_allclose(m.toarray(), (train + dev + test).toarray())


def test_split_train_test(random_graph):
    m = random_graph
    weights = [0.8, 0.2]
    train, test = split_train_test(m, weights=weights)
    sizes = np.array([train.nnz, test.nnz])

    assert_allclose(sizes / np.sum(sizes), weights, atol=0.02)
    assert_allclose(m.toarray(), (train + test).toarray())

    
def test_difference_ratio_sparse(sparse_Q1):
    sparse_Q2 = sparse_Q1.copy()
    sparse_Q2[0, 1] = -1
    assert_allclose(difference_ratio_sparse(sparse_Q1, sparse_Q2), 1/10)


def test_delete_csr_entries(sparse_Q1):
    assert sparse_Q1[0, 1] == 1
    assert sparse_Q1[0, 2] == -1
    
    m = delete_csr_entries(sparse_Q1, [(0, 1), (0, 2)])
    assert m[0, 1] == 0
    assert m[0, 2] == 0
