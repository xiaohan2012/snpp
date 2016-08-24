import random
import pytest

import numpy as np

from scipy import sparse


def make_signed_matrix(N, friends, enemies):
    Q = np.zeros((N, N))
    for i, j in friends:
        Q[i-1, j-1] = Q[j-1, i-1] = 1
    for i, j in enemies:
        Q[i-1, j-1] = Q[j-1, i-1] = -1
    for i in range(N):
        Q[i, i] = 1
        
    return Q


@pytest.fixture
def Q1():
    """a simple signed matrix
    """
    random.seed(12345)
    np.random.seed(12345)

    N = 4
    friends = [(1, 2), (3, 4)]
    enemies = [(1, 3)]
    return make_signed_matrix(N, friends, enemies)


@pytest.fixture
def sparse_Q1():
    return sparse.csr_matrix(Q1())


@pytest.fixture
def Q1_result():
    exp = np.array([[ 1.,  1.,  -1.,  -1.],
                    [ 1.,  1.,  -1.,  -1.],
                    [-1.,  -1.,  1.,  1.],
                    [-1.,  -1.,  1.,  1.]])
    return exp
