import scipy
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from pyspark.mllib.recommendation import ALS

from .spectral import predict_cluster_labels
from ..utils.matrix import indexed_entries

csr_dot = csr_matrix.dot


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


def alq_with_weight(Q, W, k, **kwargs):
    """W (weight matrix doesn't matter)
    wrapper to make interface consistant
    """
    return alq(Q, k, **kwargs)


def alq(Q, k, lambda_, max_iter,
        init_method='random',
        verbose=True):
    """
    Q: observation matrix
    k: low-rank dimension
    lambda_: regularization term weight
    """
    assert init_method in {'random', 'svd'}

    if init_method == 'random':
        X = np.random.uniform(-1, 1, (Q.shape[0], k))
        Y = np.random.uniform(-1, 1, (k, Q.shape[1]))
    elif init_method == 'svd':
        X, _, Y = np.linalg.svd(Q)
        X = X[:, :k]
        Y = np.transpose(Y[:, :k])

    W = np.sign(np.abs(Q))

    weighted_errors = []
    for ii in range(max_iter):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(
                np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(k),
                np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:, i] = np.linalg.solve(
                np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(k),
                np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        error = get_error(Q, X, Y, W)
        weighted_errors.append(error)
        if verbose:
            print('{}th iteration, weighted error{}'.format(ii, error))

    return X, Y, weighted_errors


def alq_weighted_spark(A, W, k, sc, **kwargs):
    """wrapper to make interface consistant
    """
    return alq_spark(A, k, sc, **kwargs)


def alq_spark(A, k, sc, **kwargs):
    """
    Args:
    
    - A: sign matrix (csr_matrix)
    - k: number of clusters
    - sc: the spark context
    - kwargs: parameters for ALS.train except for ratings
        https://spark.apache.org/docs/1.5.1/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.train

    Return:

    X: np.ndarray (n x k)
    Y: np.ndarray (k x n)
    """
    edges = indexed_entries(A)
    edges_rdd = sc.parallelize(edges)
    model = ALS.train(edges_rdd, rank=k, **kwargs)

    u_ft = model.userFeatures()
    p_ft = model.productFeatures()

    X = u_ft.sortByKey(ascending=True).collect()
    Y = p_ft.sortByKey(ascending=True).collect()

    X = np.array(list(zip(*X))[1])
    Y = np.transpose(np.array(list(zip(*Y))[1]))

    return X, Y


def weighted_partition_sparse(A, W, k, sc, **kwargs):
    """wrapper
    """
    return partition_sparse(A, k, sc, **kwargs)


def partition_sparse(A, k, sc, **kwargs):
    """
    Args:

    - A: sparse matrix
    - sc: spark context

    Return:

    - cluster labels
    """
    X, Y = alq_spark(A, k, sc, **kwargs)
    print(X.dtype)
    
    X = np.asarray(X, dtype=np.float16)
    Y = np.asarray(Y, dtype=np.float16)
    
    A_p = np.dot(X, Y)
    _, labels = predict_cluster_labels(A_p, k, order='desc')
    return labels
