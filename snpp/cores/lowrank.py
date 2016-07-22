import random
import numpy as np

random.seed(12345)
np.random.seed(12345)

N = 4
friends = [(1, 2), (3, 4)]
enemies = [(1, 3)]

Q = np.zeros((N, N))
for i, j in friends:
    Q[i-1, j-1] = Q[j-1, i-1] = 1
for i, j in enemies:
    Q[i-1, j-1] = Q[j-1, i-1] = -1
for i in range(N):
        Q[i, i] = 1


lambda_ = 0.1
k = 2

X, _, Y = np.linalg.svd(Q)

X = np.random.uniform(-1, 1, (N, k))
Y = np.random.uniform(-1, 1, (k, N))

W = np.sign(np.abs(Q))


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

max_iter = 20
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
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
    
print(np.sign(np.dot(X, Y)))


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
        X = np.random.uniform(-1, 1, (N, k))
        Y = np.random.uniform(-1, 1, (k, N))
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
        weighted_errors.append(get_error(Q, X, Y, W))
        if verbose:
            print('{}th iteration is completed'.format(ii))

    return X, Y, weighted_errors
