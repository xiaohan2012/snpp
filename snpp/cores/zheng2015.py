# Reference:
# Zheng, 2015, Spectral Embedding of Signed Networks


import numpy as np
from .spectral import build_laplacian_related_matrices


def build_L_sns(W):
    """
    simple normalzied signed laplacian matrix
    """
    W_p, W_n, D_p, D_n, D_hat = build_laplacian_related_matrices(W)
    return np.linalg.inv(D_hat) @ ((D_p - W_p) - (D_n - W_n))


def build_L_bns(W):
    """
    balanced normalzied signed laplacian matrix
    """
    W_p, W_n, D_p, D_n, D_hat = build_laplacian_related_matrices(W)
    return np.linalg.inv(D_hat) @ (D_p - W_p + W_n)
