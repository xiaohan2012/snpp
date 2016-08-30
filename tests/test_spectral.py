import contexts as ctx
import pytest

from scipy.sparse import csr_matrix
from snpp.cores.kunegis2010 import build_L
from snpp.cores.spectral import predict_cluster_labels, predict_cluster_labels_sparse
from sklearn.metrics import adjusted_rand_score


from data import Q1


def test_sparse(Q1):
    L = build_L(Q1)
    L_sp = csr_matrix(L)
    for order in ('asc', 'desc'):
        _, labels = predict_cluster_labels(L, k=2, order=order)
        _, labels_sp = predict_cluster_labels_sparse(L_sp, k=2, order=order)

        assert adjusted_rand_score(labels, labels_sp) == 1.0
