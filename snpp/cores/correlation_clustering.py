import numpy as np
import networkx as nx
import random

from tqdm import tqdm
from itertools import product, combinations, count

random.seed(12345)

DEBUG = False


def change_sign_to_distance(g, mapping={-1: 1, 1: -1}):
    for i, j in g.edges_iter():
        g[i][j]['weight'] = mapping[g[i][j]['sign']]
    return g

@profile
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

    # to keep the distance sum of cluster pairs
    # so no need t recompute them
    clus_dist_cache = nx.Graph()

    # at each iteration, we only need to update the distance sum
    # of pairs from clus_pairs_to_consider
    # initially, we consider all pairs
    clus_pairs_to_consider = combinations(clus.keys(), 2)
    
    for _ in tqdm(count()):
        for c1, c2 in clus_pairs_to_consider:
            cross_edges = [(n1, n2)
                           for n1, n2 in product(clus[c1], clus[c2])
                           if n1 in g.adj and n2 in g.adj[n1]]
            if cross_edges:
                clus_dist_cache.add_edge(c1, c2,
                                         weight=sum(g[n1][n2]['weight'] for n1, n2 in cross_edges))
        if clus_dist_cache.number_of_edges() > 0:  # might got clusters to merge
            new_clus = {}
            # getting cluster pair with mimimum dist_sum
            min_dist_pair = min(clus_dist_cache.edges(),
                                key=lambda e: clus_dist_cache[e[0]][e[1]]['weight'])
            min_dist_sum = clus_dist_cache[min_dist_pair[0]][min_dist_pair[1]]['weight']

            if min_dist_sum < 0.0:  # merge
                (c1, c2) = min_dist_pair
                if DEBUG:
                    print('merging {} and {}'.format(c1, c2))

                new_c, rm_c = sorted([c1, c2])
                new_clus[new_c] = clus[c1] | clus[c2]

                for c, nodes in clus.items():  # copy the resst
                    if c not in {c1, c2}:
                        new_clus[c] = clus[c]
                clus = new_clus
                clus_pairs_to_consider = [(new_c, c) for c in new_clus if c != new_c]

                # tidy the cache
                clus_dist_cache.remove_node(rm_c)
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

    # if DEBUG:
    print('sample_size {}'.format(sample_size))

    # partition the samples using `cluster_func`
    C = cluster_func(g.subgraph(samples), return_dict=True, change_sign=False,
                     **kwargs)

    # if DEBUG:
    print('partition on the samples')
    print(C)

    print("remainign nodes to assign clusters {}".format(len(remaining_nodes)))

    # assign remaining nodes to the clusters independently
    remain_nodes_clus = {}
    for n in tqdm(remaining_nodes):
        # if DEBUG:
        #     print('considering n {}'.format(n))
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

        total_cost_by_clus = sum(cost_by_clus.values())
        min_cost = - total_cost_by_clus  # singleton case
        cand_clus = -1
        
        # if DEBUG:
        #     print('min_cost {}'.format(min_cost))
        
        for c, cnodes in C.items():
            if connectable_to[c]:
                cost = (2 * cost_by_clus[c] - total_cost_by_clus)
                # if DEBUG:
                #     print('c {}'.format(c))
                #     print('cost {}'.format(cost))
                if cost < min_cost:
                    min_cost = cost
                    cand_clus = c
        if DEBUG:
            print('assinging {} to {}'.format(n, cand_clus))
        remain_nodes_clus[n] = cand_clus

    # print('remainig node clusters')
    # print(remain_nodes_clus)

    for n, c in remain_nodes_clus.items():
        if c != -1:
            C[c].add(n)

    singleton_nodes = list(filter(lambda n: remain_nodes_clus[n] == -1,
                                  remain_nodes_clus))

    print('singleton_nodes ({})'.format(len(singleton_nodes)))
    # print(singleton_nodes)

    if singleton_nodes:
        C1 = cluster_func(g.subgraph(singleton_nodes), return_dict=True, **kwargs)

        # renumbering
        for c, nodes in C1.items():
            C[len(C) + c] = nodes

    if return_dict:
        return C
    else:
        return clus_dict_to_array(g, C)
