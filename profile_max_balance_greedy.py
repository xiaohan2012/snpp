import networkx as nx
import numpy as np
from snpp.cores.max_balance import faster_greedy, greedy_g
from snpp.utils.matrix import load_sparse_csr
from snpp.cores.triangle import build_edge2edges


g = nx.read_gpickle('data/slashdot/train_graph.pkl')
test_m = load_sparse_csr('data/slashdot/test.npz')
targets = set([tuple(sorted(e))
               for e in zip(*test_m.nonzero())])

# g = g.copy()
# edge2edges = build_edge2edges(g, targets)
# faster_greedy(g, C=np.ones(g.number_of_nodes()), B=5, T=targets,
#               edge2edges=edge2edges)

greedy_g(g, C=np.ones(g.number_of_nodes()), B=5, T=targets)
