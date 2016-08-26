import numpy as np
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
    

def predict_cluster_labels(L, k, order):
    """L: the laplacian matrix
    k: the k in top-k eigen vectors

    return:
    model and predicted cluster labels
    """
    assert order in {'asc', 'desc'}
    
    w, v = np.linalg.eig(L)
    # decreasing order, WRONG?
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
