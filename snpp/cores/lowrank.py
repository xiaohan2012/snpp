import scipy
from scipy.sparse import linalg
import numpy as np


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


def get_error_sparse(Q, X, Y, W):
    return ((W.multiply(Q - np.dot(X, Y)))**2).sum()


def alq_sparse(Q, k, lambda_, max_iter,
               init_method='random',
               verbose=True):
    assert init_method in {'random', 'svd'}

    if init_method == 'random':
        X = np.random.uniform(-1, 1, (Q.shape[0], k))
        Y = np.random.uniform(-1, 1, (k, Q.shape[1]))
    elif init_method == 'svd':
        X, _, Y = scipy.sparse.linalg.svds(Q, k)

    W = scipy.sparse.csr_matrix(np.abs(Q).sign())

    QT = Q.T
    WT = scipy.sparse.csr_matrix(np.abs(QT).sign())

    n_rows, n_cols = W.shape

    weighted_errors = []
    for ii in range(max_iter):
        for u in range(n_rows):
            Wu = W[u, :].toarray().flatten()
            Wu_diag = np.diag(Wu)
            X[u] = np.linalg.solve(
                np.dot(Y, np.dot(Wu_diag, Y.T)) + lambda_ * np.eye(k),
                np.dot(Y, np.dot(Wu_diag, Q[u].toarray().T))).T
        for i in range(n_cols):
            Wi = WT[i, :].toarray().flatten()
            Wi_diag = np.diag(Wi)
            Y[:, i] = linalg.spsolve(
                np.dot(X.T, np.dot(Wi_diag, X)) + lambda_ * np.eye(k),
                np.dot(X.T, np.dot(Wi_diag, QT[i].toarray().T)))
        error = get_error_sparse(Q, X, Y, W)
        weighted_errors.append(error)
        if verbose:
            print('{}th iteration, weighted error{}'.format(ii, error))

    return X, Y, weighted_errors
