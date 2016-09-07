
# coding: utf-8

# In[ ]:

import networkx as nx
import numpy as np
import _pickle as pkl

from snpp.cores.joint_part_pred import iterative_approach
from snpp.cores.max_balance import faster_greedy
from snpp.cores.lowrank import partition_graph, alq_spark, predict_signs
from snpp.cores.budget_allocation import exponential_budget,     constant_then_exponential_budget,     linear_budget,     constant_budget
from snpp.cores.louvain import best_partition
from snpp.cores.triangle import build_edge2edges
from snpp.utils.signed_graph import fill_diagonal
from snpp.utils.edge_filter import filter_by_min_triangle_count
from snpp.utils.data import load_train_test_data

from snpp.utils.spark import sc


# In[ ]:

dataset = 'slashdot'
lambda_ = 0.1
k = 40
max_iter = 20
random_seed = 123456

recache_input = False

min_tri_count = 20


# In[ ]:

g, test_m = load_train_test_data(dataset, recache_input)

test_idx_sorted = list(map(lambda e: tuple(sorted(e)), zip(*test_m.nonzero())))


# In[ ]:

print(g.number_of_edges())
print(len(test_idx_sorted))


# In[ ]:

# make prediction based on undirectionality                                                                                            
ud_preds = []
ud_truth = []
for i, j in test_idx_sorted:
    if g.has_edge(i, j):
        ud_preds.append((i, j, g[i][j]['sign']))
        s = test_m[i, j]
        if s == 0:
            s = test_m[j, i]
        ud_truth.append((i, j, s))

print('made predictions on {} edges based on undirectionality'.format(len(ud_preds)))
print("=> accuracy is {}".format(
        len(set(ud_truth).intersection(set(ud_preds))) / len(ud_truth)))


# In[ ]:

print('removing already predicted entries')
idx_i, idx_j, data = map(list, zip(*ud_preds))
targets = set(test_idx_sorted) - set(zip(idx_i, idx_j))
targets = set([tuple(sorted(e)) for e in targets])  # sort each edge so that i <= j                                                    
print('=> remaining #targets {}'.format(len(targets)))

print('filtering edges with at least {} triangles'.format(min_tri_count))
filtered_targets = set(filter_by_min_triangle_count(g, targets, min_tri_count))
print('=> remaining #targets {}'.format(len(filtered_targets)))


# In[ ]:

# %%timeit -r 1 -n 1
# start the iterative approach
part, iter_preds, status = iterative_approach(
    g,
    T=filtered_targets,
    k=k,
    graph_partition_f=partition_graph,
    graph_partition_kwargs=dict(sc=sc,
                                lambda_=lambda_, iterations=max_iter,
                                seed=random_seed),
    budget_allocation_f=constant_budget,
    budget_allocation_kwargs=dict(const=50),
    solve_maxbalance_f=faster_greedy,
    solve_maxbalance_kwargs={'edge2edges': build_edge2edges(g.copy(),
                                                            targets)},
    truth=set([(i, j, (test_m[i, j]
                       if test_m[i, j] != 0
                       else test_m[j, i]))
               for i, j in filtered_targets]),
    perform_last_partition=False)

print('dumping result...')
pkl.dump(status, open('data/{}/status.pkl'.format(dataset), 'wb'))

print('made prediction on {} edges using iterative'.format(status.pred_cnt_list[-1]))
print('=> accuracy is {}'.format(status.acc_list[-1]))


# In[ ]:

print(g.number_of_edges())


# In[ ]:

# %%timeit -r 1 -n 1
A = nx.to_scipy_sparse_matrix(g, nodelist=g.nodes(),
                              weight='sign', format='csr')
A = fill_diagonal(A)
# assert (A.nnz - A.shape[0]) == len(targets)

X, Y = alq_spark(A, k=k, sc=sc,
                 lambda_=lambda_, iterations=max_iter,
                 seed=random_seed)


# In[ ]:

remaining_targets = targets - filtered_targets
print('predicting using lowrank method on {} edges'.format(len(remaining_targets)))
lr_preds = predict_signs(X, Y, remaining_targets, sc)


# In[ ]:

lr_preds, iter_preds = set(lr_preds), set(iter_preds)
assert len(lr_preds.intersection(iter_preds)) == 0


# In[ ]:

def nz_value(m, i, j):
    return (m[i, j] if m[i, j] != 0 else m[j, i])


# In[ ]:

truth = set((i, j, nz_value(test_m, i, j)) for i, j in test_idx_sorted)
preds = lr_preds | iter_preds | set(ud_preds)
assert len(preds) == len(truth)
print('=> undirectionality accuracy {} ({})'.format(len(truth.intersection(set(ud_preds))) / len(ud_preds), len(ud_preds)))
print('=> iteractive accuracy {} ({})'.format(len(truth.intersection(iter_preds)) / len(iter_preds), len(iter_preds)))
print('=> lowrank accuracy {} ({})'.format(len(truth.intersection(lr_preds)) / len(lr_preds), len(lr_preds)))
print('=> final accuracy {}'.format(len(truth.intersection(preds)) / len(truth)))

