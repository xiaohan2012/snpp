import contexts as ctx
import pytest

from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from snpp.cores.kunegis2010 import build_L
from snpp.cores.spectral import predict_cluster_labels, predict_cluster_labels_sparse, \
    predict_cluster_labels_svd
from sklearn.metrics import adjusted_rand_score


def test_sparse(Q1):
    L = build_L(Q1)
    L_sp = csr_matrix(L)
    for order in ('asc', 'desc'):
        _, labels = predict_cluster_labels(L, k=2, order=order)
        _, labels_sp = predict_cluster_labels_sparse(L_sp, k=2, order=order)

        assert adjusted_rand_score(labels, labels_sp) == 1.0
        

def test_svd(Q1):
    L = build_L(Q1)
    _, labels = predict_cluster_labels_svd(L, k=2, order='asc')
    assert adjusted_rand_score([0, 0, 1, 1], labels) == 1.0

