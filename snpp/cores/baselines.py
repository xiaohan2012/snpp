import numpy as np
from itertools import combinations


def predict_signs_via_partition(cluster_labels):
    """given clustering labels,
    predict the sign matrix
    """
    N = len(cluster_labels)
    pred_sign_mat = np.diag(np.ones(N))
    for i, j in combinations(list(range(N)), 2):
        if cluster_labels[i] == cluster_labels[j]:
            v = 1
        else:
            v = -1
        pred_sign_mat[i, j] = pred_sign_mat[j, i] = v
    return pred_sign_mat
