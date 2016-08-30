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
from snpp.cores.zheng2015 import build_L_sns as build_zheng2015_sns, \
    build_L_bns as build_zheng2015_bns
from snpp.cores.kunegis2010 import build_L as build_kunegis2010


random.seed(12345)
np.random.seed(12345)


def do_one_run(group_size,
               group_number,
               known_edge_percentage):
    Q, true_Q = example_for_intuition(group_size,
                                      group_number,
                                      known_edge_percentage)
    N = group_size * group_number
    
    k = group_number
    lambda_ = 0.2
    max_iter = 20

    # low-rank (matrix completion) method
    X, Y, errors = alq(Q, k, lambda_, max_iter,
                       init_method='svd',
                       verbose=False)

    pred_Q = np.sign(np.dot(X, Y))n
    # print('true_Q: \n{}'.format(true_Q))
    # print('pred_Q: \n{}'.format(pred_Q))

    # accuracy of completed matrix
    lr_acc = np.count_nonzero(true_Q == pred_Q) / (N * N)
    Q_p = np.dot(X, Y)
    masked_pred_Q = np.array(np.sign(Q_p) == 1,
                             dtype=np.int)

    def evaluate(input_matrix, eigen_order):
        _, pred_cluster_labels = predict_cluster_labels(
            input_matrix, k, eigen_order)
        true_cluster_labels = [j for i in range(group_number)
                               for j in repeat(i, group_size)]

        # print('true_cluster_labels:')
        # print(true_cluster_labels)
        # print('pred_cluster_labels:')
        # print(pred_cluster_labels)
        arc = adjusted_rand_score(true_cluster_labels, pred_cluster_labels)

        # partition-based sign prediction
        pred_sign_mat = predict_signs_via_partition(pred_cluster_labels)
        p_acc = np.count_nonzero(true_Q == pred_sign_mat) / (N * N)
        return arc, p_acc

    # using approximated sign matrix of low-rank method
    lr_arc, lr_p_acc = evaluate(Q_p, 'desc')
    
    # using zheng2015_sns
    L_zheng2015_sns = build_zheng2015_sns(Q)
    zheng2015_sns_arc, zheng2015_sns_p_acc = evaluate(L_zheng2015_sns, 'asc')
    
    # using zheng2015_bns
    L_zheng2015_bns = build_zheng2015_bns(Q)
    zheng2015_bns_arc, zheng2015_bns_p_acc = evaluate(L_zheng2015_bns, 'asc')

    # using kunegis2010
    L_kunegis2010 = build_kunegis2010(Q)
    kunegis2010_arc, kunegis2010_p_acc = evaluate(L_kunegis2010, 'asc')
    
    return {
        # lowrank
        'accuracy(lowrank)': lr_acc,
        'accuracy(lowrank + partition)': lr_p_acc,
        'adjusted_rand_score(lowrank)': lr_arc,
        # zheng2015_sns
        'accuracy(zheng2015_sns + partition)': zheng2015_sns_p_acc,
        'adjusted_rand_score(zheng2015_sns)': zheng2015_sns_arc,
        # zheng2015_bns
        'accuracy(zheng2015_bns + partition)': zheng2015_bns_p_acc,
        'adjusted_rand_score(zheng2015_bns)': zheng2015_bns_arc,
        # kunegis2010
        'accuracy(kunegis2010 + partition)': kunegis2010_p_acc,
        'adjusted_rand_score(kunegis2010)': kunegis2010_arc
    }


def do_n_times(run_times,
               parameters,  # dict
               variable_name,
               domain):  # list
    columns = ['run_id',
               'accuracy(lowrank)', 'adjusted_rand_score(lowrank)',
               'accuracy(lowrank + partition)',
               'accuracy(zheng2015_sns + partition)',
               'adjusted_rand_score(zheng2015_sns)',
               'accuracy(zheng2015_bns + partition)',
               'adjusted_rand_score(zheng2015_bns)',
               'accuracy(kunegis2010 + partition)',
               'adjusted_rand_score(kunegis2010)',
               'group_size', 'group_number', 'known_edge_percentage']
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

    accuracy_columns = list(filter(lambda c: c.startswith('acc'), columns))
    arc_columns = list(filter(lambda c: c.startswith('adj'), columns))
    
    ax = stat[accuracy_columns].plot(style=['o'])
    fig = ax.get_figure()
    fig.savefig(
        'figures/motivation/{}-accuracy.png'.format(variable_name))

    ax = stat[arc_columns].plot()
    fig = ax.get_figure()
    fig.savefig(
        'figures/motivation/{}-adjusted_rand_score.png'.format(variable_name))
    
    print('\n')
    

def main():
    run_times = 10
    do_n_times(run_times,
               {'group_size': 4,
                'group_number': 10},
               'known_edge_percentage',
               np.linspace(0.1, 0.96, num=10))
    
    # do_n_times(run_times,
    #            {'group_size': 4,
    #             'known_edge_percentage': 0.5},
    #            'group_number',
    #            list(range(1, 20)))
    # do_n_times(run_times,
    #            {'group_number': 4,
    #             'known_edge_percentage': 0.5},
    #            'group_size',
    #            list(range(1, 15)))


# if __name__ == '__main__':
main()
