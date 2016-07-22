import networkx as nx
from snpp.cores.lowrank import alq

dataset = 'epinions'
method = 'lowrank'
lambda_ = 0.2
k = 10
max_iter = 100

g = nx.read_gpickle('data/epinions.pkl')

# 1. make g undirected
# 2. partition the edges into train and test (what's the ratio? Jure's paper)
# 3. run algorithm (coping with sparse data)
# 4. evaluate accuracy, f1 (Jure's paper)

Q = nx.adjacency_matrix(g)
alq(Q, k, lambda_, max_iter,
    init_method='random',
    verbose=True)

