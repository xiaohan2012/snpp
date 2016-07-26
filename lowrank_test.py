import numpy as np
import random
from itertools import combinations
from snpp.cores.lowrank import alq

random.seed(12345)
np.random.seed(12345)


def do_one_run(group_size,
               n_groups,
               foe_number_per_pair):

    N = group_size * n_groups
    Q = np.zeros((N, N))
    true_Q = np.zeros((N, N))

    friends = []
    foes = []
    members_by_group = [
        list(
            range(g * group_size, g * group_size + group_size))
        for g in range(n_groups)]
    
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


    k = n_groups
    lambda_ = 0.2
    max_iter = 20

    X, Y, errors = alq(Q, k, lambda_, max_iter,
                       init_method='random',
                       verbose=False)

    pred_Q = np.sign(np.dot(X, Y))
    # print('true_Q: \n{}'.format(true_Q))
    # print('pred_Q: \n{}'.format(pred_Q))

    acc = np.count_nonzero(true_Q == pred_Q) / (N * N)
    masked_pred_Q = np.array(np.sign(np.dot(X, Y)) == 1,
                             dtype=np.int)
    return acc

foe_counts = list(range(1, 10))
acc_list = [do_one_run(4, 10, foe_number_per_pair=i)
            for i in foe_counts]
print('foe number vs accuracy\n{}\n{}'.format(foe_counts,
                                              acc_list))


group_numbers = list(range(1, 10))
acc_list = [do_one_run(4, i, 1)
            for i in group_numbers]
print('group_number vs accuracy\n{}\n{}'.format(group_numbers,
                                                acc_list))

group_sizes = list(range(1, 10))
acc_list = [do_one_run(i, 3, 1)
            for i in group_sizes]
print('group_size vs accuracy\n{}\n{}'.format(group_sizes,
                                              acc_list))
