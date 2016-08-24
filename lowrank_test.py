import matplotlib

import random
import numpy as np
import pandas as pd

from itertools import repeat
from copy import copy

from sklearn.metrics.cluster import adjusted_rand_score

from snpp.cores.lowrank import alq
from snpp.cores.spectral import predict_cluster_labels
from snpp.utils.data import example_for_intuition
from snpp.cores.baselines import predict_signs_via_partition
from snpp.cores.zheng2015 import build_L_sns


random.seed(12345)
np.random.seed(12345)


def do_one_run(group_size,
               group_number,
               foe_number_per_pair):
    Q, true_Q = example_for_intuition(group_size,
                                      group_number,
                                      foe_number_per_pair)
    N = group_size * group_number
    
    k = group_number
    lambda_ = 0.2
    max_iter = 20

    X, Y, errors = alq(Q, k, lambda_, max_iter,
                       init_method='svd',
                       verbose=False)

    pred_Q = np.sign(np.dot(X, Y))
    # print('true_Q: \n{}'.format(true_Q))
    # print('pred_Q: \n{}'.format(pred_Q))

    lr_acc = np.count_nonzero(true_Q == pred_Q) / (N * N)
    Q_p = np.dot(X, Y)
    masked_pred_Q = np.array(np.sign(Q_p) == 1,
                             dtype=np.int)

    def evaluate_spectral_partition_and_prediction(input_matrix):
        _, pred_cluster_labels = predict_cluster_labels(input_matrix, k)
        true_cluster_labels = [j for i in range(group_number)
                               for j in repeat(i, group_size)]

        arc = adjusted_rand_score(true_cluster_labels, pred_cluster_labels)

        # partition-based sign prediction
        pred_sign_mat = predict_signs_via_partition(pred_cluster_labels)
        p_acc = np.count_nonzero(true_Q == pred_sign_mat) / (N * N)
        return arc, p_acc

    # using approximated sign matrix
    lr_arc, lr_p_acc = evaluate_spectral_partition_and_prediction(Q_p)
    
    # using signed spectral method \cite{zheng2015}
    L = build_L_sns(Q)
    spec_arc, spec_p_acc = evaluate_spectral_partition_and_prediction(L)
    return {
        'accuracy(lowrank)': lr_acc,
        'adjusted_rand_score(lowrank)': lr_arc,
        'accuracy(partition)': lr_p_acc,
        'accuracy(spectral)': spec_p_acc,
        'adjusted_rand_score(spectral)': spec_arc}


def do_n_times(run_times,
               parameters,  # dict
               variable_name,
               domain):  # list
    columns = ['run_id',
               'accuracy(lowrank)', 'adjusted_rand_score(lowrank)',
               'accuracy(partition)',
               'accuracy(spectral)', 'adjusted_rand_score(spectral)',
               'group_size', 'group_number', 'foe_number_per_pair']
    results = []
    for i in range(run_times):
        print('run {}'.format(i))
        for j in domain:
            parameters[variable_name] = j
            eval_result = do_one_run(**parameters)

            row = copy(parameters)
            row.update({'run_id': i})
            row.update(eval_result)

            results.append(row)

    eval_metrics = list(eval_result.keys())
    df = pd.DataFrame(results, columns=columns)
    stat = df.groupby(variable_name)[eval_metrics].mean()
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
