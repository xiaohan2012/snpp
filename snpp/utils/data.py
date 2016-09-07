import random
import pandas as pd
import numpy as np
import networkx as nx

from scipy.sparse import dok_matrix

from snpp.utils.matrix import load_sparse_csr, \
    save_sparse_csr, \
    split_train_test
from snpp.utils.signed_graph import matrix2graph
    
from itertools import combinations


def load_csv_network(path):
    df = pd.read_csv(path, sep='\t')
    g = nx.DiGraph()
    g.name = path

    g.add_edges_from(
        (r['FromNodeId'], r['ToNodeId'], {'sign': r['Sign']})
        for i, r in df.iterrows()
    )
    return g


def load_csv_as_sparse(path):
    """node ids should go from 0 to N
    """
    df = pd.read_csv(path, sep='\t')
    nodes = set(df['FromNodeId'].unique()).union(set(df['ToNodeId'].unique()))
    N = max(nodes) + 1  # DOUBT
    m = dok_matrix((N, N))
    for i, r in df.iterrows():
        m[r['FromNodeId'], r['ToNodeId']] = r['Sign']

    return m.tocsr()


def example_for_intuition(group_size, group_number, known_edge_percentage):
    assert known_edge_percentage <= 1
    N = group_size * group_number
    mat_size = N * (N - 1) / 2

    Q = np.zeros((N, N))
    true_Q = np.zeros((N, N))

    members_by_group = [
        list(
            range(g * group_size, g * group_size + group_size))
        for g in range(group_number)]
    
    for g in members_by_group:
        for i in g:
            true_Q[i, i] = 1
        for i, j in combinations(g, 2):
            true_Q[i, j] = true_Q[j, i] = 1

    true_Q[true_Q == 0] = -1

    n_known = int(mat_size * known_edge_percentage)

    all_edges = list(combinations(range(N), 2))

    try:
        sel_idx = np.random.choice(len(all_edges), size=n_known, replace=False)
    except ValueError:  # ValueError: Cannot take a larger sample than population when 'replace=False'
        sel_idx = np.random.choice(len(all_edges), size=len(all_edges), replace=False)

    for idx in sel_idx:
        i, j = all_edges[idx]
        Q[i, j] = Q[j, i] = true_Q[i, j]

    for i in range(N):
        Q[i, i] = 1

    return Q, true_Q


def make_lowrank_matrix(g_size, rank):
    _, M = example_for_intuition(g_size, rank, 0.0)
    return M


def load_train_test_graphs(dataset, recache_input):
    raw_mat_path = 'data/{}.npz'.format(dataset)
    train_graph_path = 'data/{}/train_graph.pkl'.format(dataset)
    test_graph_path = 'data/{}/test_graph.pkl'.format(dataset)

    if recache_input:
        print('loading sparse matrix from {}'.format(raw_mat_path))
        m = load_sparse_csr(raw_mat_path)

        print('splitting train and test...')
        train_m, test_m = split_train_test(
            m,
            weights=[0.9, 0.1])

        print('converting to nx.DiGraph')
        train_g = nx.from_scipy_sparse_matrix(train_m, create_using=nx.DiGraph(), edge_attribute='sign')
        test_g = nx.from_scipy_sparse_matrix(test_m, create_using=nx.DiGraph(), edge_attribute='sign')
                
        print('saving train and test graphs...')
        nx.write_gpickle(train_g, train_graph_path)
        nx.write_gpickle(test_g, test_graph_path)
    else:
        print('loading train and test graphs...')
        train_g = nx.read_gpickle(train_graph_path)
        test_g = nx.read_gpickle(test_graph_path)
    return train_g, test_g


def make_signed_matrix(N, friends, enemies):
    Q = np.zeros((N, N))
    for i, j in friends:
        Q[i-1, j-1] = Q[j-1, i-1] = 1
    for i, j in enemies:
        Q[i-1, j-1] = Q[j-1, i-1] = -1
    for i in range(N):
        Q[i, i] = 1
        
    return Q
