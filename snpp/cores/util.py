import numpy as np


def build_laplacian_related_matrices(W):
    """W: the sign matrix
    """
    W_p = np.maximum(W, 0)
    W_n = - np.minimum(W, 0)
    D_p = np.diag(np.sum(W_p, axis=1))
    D_n = np.diag(np.sum(W_n, axis=1))
    D_hat = D_p + D_n
    return W_p, W_n, D_p, D_n, D_hat
