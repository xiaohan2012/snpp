# Reference:
# Kunegis, Spectral Analysis of Signed Graphs for Clustering, Prediction and Visualization, 2010


from .spectral import build_laplacian_related_matrices


def build_L(W):
    W_p, W_n, D_p, D_n, D_hat = build_laplacian_related_matrices(W)
    return D_p + D_n - W_p + W_n
