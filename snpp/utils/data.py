import random
import pandas as pd
import numpy as np
import networkx as nx


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


def example_for_intuition(group_size, group_number, foe_number_per_pair):
    N = group_size * group_number
    Q = np.zeros((N, N))
    true_Q = np.zeros((N, N))

    friends = []
    foes = []
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

    for g in members_by_group:
        for i in g[:-1]:
            friends.append((i, i + 1))
            
    for g1, g2 in combinations(members_by_group, 2):
        for i in range(foe_number_per_pair):
            foe = (random.choice(g1), random.choice(g2))
            foes.append(foe)

    for i in range(N):
        Q[i, i] = 1

    for u, v in friends:
        Q[u, v] = Q[v, u] = 1

    for u, v in foes:
        Q[u, v] = Q[v, u] = -1

    return Q, true_Q


def main():
    dataset = 'epinions'
    path = 'data/soc-sign-{}.txt'.format(dataset)
    g = load_csv_network(path)
    nx.write_gpickle(g, 'data/{}.pkl'.format(dataset))


if __name__ == '__main__':
    main()
