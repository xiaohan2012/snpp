import numpy as np
import scipy
from scipy.sparse import  issparse
from sklearn.cluster import KMeans


def build_laplacian_related_matrices(W):
    """W: the sign matrix
    """
    W_p = np.maximum(W, 0)
    W_n = - np.minimum(W, 0)
    D_p = np.diag(np.sum(W_p, axis=1))
    D_n = np.diag(np.sum(W_n, axis=1))
    D_hat = D_p + D_n
    return W_p, W_n, D_p, D_n, D_hat


def build_laplacian_related_matrices_sparse(W):
    """W: the sign matrix (sparse)
    """
    W_p = np.maximum(W, 0)
    W_n = - np.minimum(W, 0)
    D_p = np.diag(np.sum(W_p, axis=1))
    D_n = np.diag(np.sum(W_n, axis=1))
    D_hat = D_p + D_n
    return W_p, W_n, D_p, D_n, D_hat


def predict_cluster_labels_svd(M, k, order):
    """
    for non-square matrices

    M: mxn matrix, for exmample the Laplacian
    """
    U, s, vh = scipy.linalg.svd(M, full_matrices=False)
    print('eigen values: {}'.format(s[:k]))
    if order == 'desc':
        X = U[:, :k]
    else:
        X = U[:, -k:]
    
    model = KMeans(n_clusters=k)
    pred_y = model.fit_predict(X)
    return model, pred_y


def predict_cluster_labels(L, k, order):
    """L: the laplacian matrix
    k: the k in top-k eigen vectors

    return:
    model and predicted cluster labels
    """
    assert order in {'asc', 'desc'}
    
    w, v = np.linalg.eig(L)
    
    if order == 'desc':
        indx = np.argsort(w)[::-1]
    else:
        indx = np.argsort(w)
    w = w[indx]
    v = v[:, indx]
    X = v[:, :k]

    model = KMeans(n_clusters=k)
    pred_y = model.fit_predict(X)
    return model, pred_y


def predict_cluster_labels_sparse(L, k, order, **kwargs):
    """L: the laplacian matrix (sparse)
    k: the k in top-k eigen vectors

    return:
    model and predicted cluster labels
    """
    assert order in {'asc', 'desc'}
    assert issparse(L)

    m = {'asc': 'SM',
         'desc': 'LM'}
    # SVD
    u, s, vt = scipy.sparse.linalg.svds(L, k=k, which=m[order], **kwargs)
    
    X = u
    model = KMeans(n_clusters=k)
    pred_y = model.fit_predict(X)
    return model, pred_y
