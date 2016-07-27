import matplotlib
matplotlib.style.use('ggplot')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations, repeat
from copy import copy

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

from snpp.cores.lowrank import alq

random.seed(12345)
np.random.seed(12345)


def do_one_run(group_size,
               group_number,
               foe_number_per_pair):

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

    k = group_number
    lambda_ = 0.2
    max_iter = 20

    X, Y, errors = alq(Q, k, lambda_, max_iter,
                       init_method='svd',
                       verbose=False)

    pred_Q = np.sign(np.dot(X, Y))
    # print('true_Q: \n{}'.format(true_Q))
    # print('pred_Q: \n{}'.format(pred_Q))

    acc = np.count_nonzero(true_Q == pred_Q) / (N * N)
    Q_p = np.dot(X, Y)
    masked_pred_Q = np.array(np.sign(Q_p) == 1,
                             dtype=np.int)


    # clustering
    w, v = np.linalg.eig(Q_p)
    indx = np.argsort(w)[::-1]
    w = w[indx]
    v = v[:, indx]
    X = v[:, :k]
    # print(X)
    model = KMeans(n_clusters=k)
    pred_y = model.fit_predict(X)
    true_y = [j for i in range(group_number) for j in repeat(i, group_size)]
    # print(pred_y)
    arc = adjusted_rand_score(true_y, pred_y)
    # print('adjusted_rand_score: {}'.format(arc))

    # partition-based sign prediction
    pred_sign_mat = np.diag(np.ones(N))
    for i, j in combinations(list(range(N)), 2):
        if pred_y[i] == pred_y[j]:
            v = 1
        else:
            v = -1
        pred_sign_mat[i, j] = pred_sign_mat[j, i] = v
    p_acc = np.count_nonzero(true_Q == pred_sign_mat) / (N * N)
    return acc, arc, p_acc


def do_n_times(run_times,
               parameters,  # dict
               variable_name,
               domain):  # list
    columns = ['run_id',
               'accuracy(lowrank)', 'accuracy(partition)',
               'adjusted_rand_score',
               'group_size', 'group_number', 'foe_number_per_pair']
    result = []
    for i in range(run_times):
        print('run {}'.format(i))
        for j in domain:
            parameters[variable_name] = j
            acc, arc, p_acc = do_one_run(**parameters)
            row = copy(parameters)
            row.update({'run_id': i,
                        'accuracy(lowrank)': acc,
                        'accuracy(partition)': p_acc,
                        'adjusted_rand_score': arc})
            result.append(row)
            
    df = pd.DataFrame(result, columns=columns)
    stat = df.groupby(variable_name)[['accuracy(lowrank)',
                                      'accuracy(partition)',
                                      'adjusted_rand_score']].mean()
    print(stat)
    ax = stat.plot()
    fig = ax.get_figure()
    fig.savefig('figures/lowrank_test/{}.png'.format(variable_name))
    print('\n')
    

def main():
    run_times = 10
    do_n_times(run_times,
               {'group_size': 4,
                'group_number': 10},
               'foe_number_per_pair',
               list(range(1, 20)))

    do_n_times(run_times,
               {'group_size': 4,
                'foe_number_per_pair': 1},
               'group_number',
               list(range(1, 20)))
    do_n_times(run_times,
               {'group_number': 4,
                'foe_number_per_pair': 1},
               'group_size',
               list(range(1, 15)))


# if __name__ == '__main__':
main()
