import scipy
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import numpy as np

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


def get_error_sparse(Q, X, Y, W):
    return ((W.multiply(Q - np.dot(X, Y)))**2).sum()


# @profile
def alq_sparse(Q, k, lambda_, max_iter,
               init_method='random',
               verbose=True):
    assert init_method in {'random', 'svd'}

    if init_method == 'random':
        X = np.random.uniform(-1, 1, (Q.shape[0], k))
        Y = np.random.uniform(-1, 1, (k, Q.shape[1]))
    elif init_method == 'svd':
        X, _, Y = sparse.linalg.svds(Q, k)
        Y = Y.T

    W = sparse.csr_matrix(np.abs(Q).sign())

    QT = Q.T
    WT = sparse.csr_matrix(np.abs(QT).sign())

    n_rows, n_cols = W.shape

    eye_k_by_lambda = lambda_ * sparse.eye(k)
    weighted_errors = []
    for ii in range(max_iter):
        for u in range(n_rows):
            if verbose and u % 100 == 0:
                print('inside loop: u={}'.format(u))
            if u > 500:
                break
            Wu = W[u, :].toarray().flatten()
            Wu_diag = sparse.diags(Wu, format='csr')
            # print('type(Y) = {}', type(Y))
            # print('type(Y.T) = {}', type(Y.T))
            # print('type(Wu_diag) = {}', type(Wu_diag))
            # print('type(Q[u].T) = {}', type(Q[u].T))
            print(csr_dot(Wu_diag, Y.T))
            X[u] = np.linalg.solve(
                csr_dot(Y, csr_dot(Wu_diag, Y.T)) + eye_k_by_lambda,
                Y @ Wu_diag @ Q[u].T).T

        break
    
        for i in range(n_cols):
            if verbose and i % 100 == 0:
                print('inside loop: i={}'.format(i))
            Wi = WT[i, :].toarray().flatten()
            Wi_diag = sparse.diags(Wi, format='csr')
            Y[:, i] = linalg.spsolve(
                X.T @ Wi_diag @ X + eye_k_by_lambda,
                X.T @ Wi_diag @ QT[i].T)
        error = get_error_sparse(Q, X, Y, W)
        weighted_errors.append(error)
        if verbose:
            print('{}th iteration, weighted error {}'.format(ii, error))

    return X, Y, weighted_errors
