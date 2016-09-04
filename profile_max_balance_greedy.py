import numpy as np
from snpp.cores.max_balance import greedy
from snpp.utils.matrix import load_sparse_csr


train_m = load_sparse_csr('data/slashdot/train_sym.npz')
test_m = load_sparse_csr('data/slashdot/test.npz')
targets = set(zip(*test_m.nonzero()))

n, _ = train_m.shape
print('what the...')
greedy_g(train_m, None, C=np.ones(n), B=5, T=targets)
