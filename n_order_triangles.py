# calculate the statistics on n-order triangles

from tqdm import tqdm
import itertools
from collections import Counter
from snpp.utils.matrix import load_sparse_csr, \
    split_train_test
from snpp.utils.signed_graph import matrix2graph


dataset = 'slashdot'
raw_mat_path = 'data/{}.npz'.format(dataset)
random_seed = 123456

m = load_sparse_csr(raw_mat_path)

print('split_train_test')
train_m, test_m = split_train_test(
    m,
    weights=[0.9, 0.1])

test_entries = set(tuple(sorted((i, j)))
                   for i, j in zip(*test_m.nonzero()))
g = matrix2graph(m, None)

# getting triangles
nodes_nbrs = g.adj.items()

triangles = set()
for v, v_nbrs in tqdm(nodes_nbrs):
    vs = set(v_nbrs) - set([v])
    ntriangles = 0
    for w in vs:
        ws = set(g[w]) - set([w])
        for u in vs.intersection(ws):
            triangles.add(tuple(sorted([u, v, w])))
        
print("{} triangles".format(len(triangles)))


cnt = Counter()
for t in triangles:
    its = filter(lambda e: tuple(sorted(e)) in test_entries,
                 itertools.combinations(t, 2))
    cnt[len(list(its))] += 1

print(cnt)
