import scipy
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from pyspark.mllib.recommendation import ALS

csr_dot = csr_matrix.dot


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


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


def alq_spark(edges, **kwargs):
    """
    Args:
    
    - edges: RDD of (node_1, node_2, sign)
    - kwargs: parameters for ALS.train except for ratings
        https://spark.apache.org/docs/1.5.1/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.train

    Return:

    X: np.ndarray (n x k)
    Y: np.ndarray (k x n)
    """
    model = ALS.train(edges, **kwargs)
    X = model.userFeatures().sortByKey(ascending=True).collect()
    Y = model.productFeatures().sortByKey(ascending=True).collect()

    X = np.array(list(zip(*X))[1])
    Y = np.transpose(np.array(list(zip(*Y))[1]))

    return X, Y
    
