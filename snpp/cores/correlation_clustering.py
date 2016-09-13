import numpy as np
import random
from tqdm import tqdm
from itertools import product

DEBUG = True


def change_sign_to_distance(g, mapping={-1: 1, 1: -1}):
    for i, j in g.edges_iter():
        g[i][j]['weight'] = mapping[g[i][j]['sign']]
    return g


def agglomerative(g, return_dict=True, change_sign=False):
    """
    return_dict: return dict if True, otherwise return label array
    """
    if g.number_of_nodes() == 0:
        raise ValueError('empty graph')

    if change_sign:
        g = change_sign_to_distance(g.copy())
    
    clus = {i: {n}
            for i, n in enumerate(g.nodes_iter())}

    while True:
        found_candidate = False
        min_dist_sum = float('inf')
        candidate_clus_pair = None
        for c1, c1_nodes in clus.items():
            for c2, c2_nodes in clus.items():
                if c1 == c2:
                    continue
                cross_edges = [(n1, n2)
                               for n1, n2 in product(c1_nodes, c2_nodes)
                               if g.has_edge(n1, n2)]
                if cross_edges:
                    dist_sum = sum(g[n1][n2]['weight'] for n1, n2 in cross_edges)
                    if dist_sum < min_dist_sum:
                        min_dist_sum = dist_sum
                        candidate_clus_pair = (c1, c2)
                        num_cross_edges = len(cross_edges)
                        found_candidate = True
        if found_candidate:
            mean_dist = min_dist_sum / num_cross_edges
            if DEBUG:
                print("mean_dist {}".format(mean_dist))

            new_clus = {}
            if mean_dist <= 0.0:  # merge
                (c1, c2) = candidate_clus_pair
                new_clus[0] = clus[c1] | clus[c2]
                for i, (c, nodes) in enumerate(clus.items()):
                    if c not in {c1, c2}:
                        new_clus[i+1] = clus[c]
                if DEBUG:
                    print('new clustering {}'.format(new_clus))
                clus = new_clus
            else:
                if DEBUG:
                    print("no more clusters to merge")
                break
        else:
            if DEBUG:
                print('didn\'t find mergable cluster pair')
            break

    if return_dict:
        return clus
    else:
        return clus_dict_to_array(g, clus)


def clus_dict_to_array(g, clus):
    labels = np.zeros(g.number_of_nodes())
    print(labels.shape)
    for c, nodes in clus.items():
        for n in nodes:
            labels[n] = c
    return labels
    
    
def sampling_wrapper(g, cluster_func, sample_size=None, samples=None, return_dict=True, **kwargs):
    g = change_sign_to_distance(g.copy())

    # take samples
    if samples is None:
        assert sample_size > 0
        samples = random.sample(g.nodes(), sample_size)
    else:
        sample_size = len(samples)
    remaining_nodes = set(g.nodes()) - set(samples)
    
    # partition the samples using `cluster_func`
    C = cluster_func(g.subgraph(samples), return_dict=True, change_sign=False,
                     **kwargs)

    if DEBUG:
        print('partition on the samples')
        print(C)
    
    # assign remaining nodes to the clusters independently
    remain_nodes_clus = {}
    for n in tqdm(remaining_nodes):
        if DEBUG:
            print('considering n {}'.format(n))
        cost_by_clus = {}
        connectable_to = {}
        for c, cnodes in C.items():
            cost_by_clus[c] = sum(g[n][cn]['weight']
                                  for cn in cnodes
                                  if g.has_edge(n, cn))
            neg_weight_sum = sum(g[n][cn]['weight']
                                 for cn in cnodes
                                 if (g.has_edge(n, cn) and
                                     g[n][cn]['weight'] < 0))
            connectable_to[c] = (neg_weight_sum < 0)

        print(cost_by_clus)
        total_cost_by_clus = sum(cost_by_clus.values())
        min_cost = - total_cost_by_clus  # singleton case
        cand_clus = -1
        
        if DEBUG:
            print('min_cost {}'.format(min_cost))
        
        for c, cnodes in C.items():
            if connectable_to[c]:
                cost = (2 * cost_by_clus[c] - total_cost_by_clus)
                if DEBUG:
                    print('c {}'.format(c))
                    print('cost {}'.format(cost))
                if cost < min_cost:
                    min_cost = cost
                    cand_clus = c
        if DEBUG:
            print('assinging {} to {}'.format(n, cand_clus))
        remain_nodes_clus[n] = cand_clus

    if DEBUG:
        print('remainig node clusters')
        print(remain_nodes_clus)

    for n, c in remain_nodes_clus.items():
        if c != -1:
            C[c].add(n)

    singleton_nodes = list(filter(lambda n: remain_nodes_clus[n] == -1,
                                  remain_nodes_clus))
    if DEBUG:
        print('singleton_nodes')
        print(singleton_nodes)

    if singleton_nodes:
        C1 = cluster_func(g.subgraph(singleton_nodes), return_dict=True, **kwargs)

        # renumbering
        for c, nodes in C1.items():
            C[len(C) + c] = nodes

    if return_dict:
        return C
    else:
        return clus_dict_to_array(g, C)
