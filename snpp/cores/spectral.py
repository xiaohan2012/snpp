import numpy as np
from sklearn.cluster import KMeans
    

def predict_cluster_labels(L, k):
    """L: the laplacian matrix
    k: the k in top-k eigen vectors

    return:
    model and predicted cluster labels
    """
    w, v = np.linalg.eig(L)
    indx = np.argsort(w)[::-1]
    w = w[indx]
    v = v[:, indx]
    X = v[:, :k]

    model = KMeans(n_clusters=k)
    pred_y = model.fit_predict(X)
    return model, pred_y
