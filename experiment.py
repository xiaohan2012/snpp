from pyspark.sql import SparkSession

import networkx as nx
from scipy.sparse import coo_matrix, dok_matrix
import _pickle as pkl

from snpp.cores.joint_part_pred import iterative_approach
from snpp.cores.max_balance import faster_greedy
from snpp.cores.lowrank import partition_graph
from snpp.cores.budget_allocation import exponential_budget, \
    constant_then_exponential_budget, \
    linear_budget
from snpp.cores.louvain import best_partition
from snpp.cores.triangle import build_edge2edges
from snpp.utils.matrix import load_sparse_csr, \
    save_sparse_csr, \
    split_train_test, \
    difference_ratio_sparse
from snpp.utils.status import Status
from snpp.utils.signed_graph import matrix2graph


dataset = 'slashdot'
method = 'lowrank'
lambda_ = 0.1
k = 40
max_iter = 20
random_seed = 123456

raw_mat_path = 'data/{}.npz'.format(dataset)
train_graph_path = 'data/{}/train_graph.pkl'.format(dataset)
test_data_path = 'data/{}/test'.format(dataset)

recache_input = False


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Signed Network Experiment")\
        .getOrCreate()
    spark_context = spark.sparkContext
    spark_context.setCheckpointDir('.checkpoint')  # stackoverflow errors

    if recache_input:
        print('loading sparse matrix from {}'.format(raw_mat_path))
        m = load_sparse_csr(raw_mat_path)

        print('splitting train and test...')
        train_m, test_m = split_train_test(
            m,
            weights=[0.9, 0.1])

        nodes = list(range(m.shape[0]))
        print('converting to nx.Graph')
        g = matrix2graph(train_m, W=None,
                         nodes=nodes, multigraph=False)
                
        print('saving training graph and test data...')
        nx.write_gpickle(g, train_graph_path)
        save_sparse_csr(test_data_path, test_m)
    else:
        print('loading pre-split train and test matrix...')
        g = nx.read_gpickle(train_graph_path)
        test_m = load_sparse_csr(test_data_path + '.npz')

    # make prediction based on undirectionality
    preds = []
    truth = []
    for i, j in zip(*test_m.nonzero()):
        if g.has_edge(i, j):
            preds.append((i, j, g[i][j]['sign']))
            truth.append((i, j, test_m[i, j]))

    print('Made predictions on {} edges based on undirectionality'.format(len(preds)))
    print("Accuracy is {}".format(
        len(set(truth).intersection(set(preds))) / len(truth)))

    # remove already predicted entries
    idx_i, idx_j, data = map(list, zip(*preds))
    targets = set(zip(*test_m.nonzero())) - set(zip(idx_i, idx_j))
    # sort each edge so that i <= j
    targets = set([tuple(sorted(e)) for e in targets])
    print('remaining #targets {}'.format(len(targets)))
    
    # start the iterative approach
    part, predictions, status = iterative_approach(
        g,
        T=targets,
        k=k,
        graph_partition_f=partition_graph,
        graph_partition_kwargs=dict(sc=spark_context,
                                    lambda_=lambda_, iterations=max_iter,
                                    seed=random_seed),
        budget_allocation_f=linear_budget,
        budget_allocation_kwargs=dict(linear_const=100),
        solve_maxbalance_f=faster_greedy,
        solve_maxbalance_kwargs={'edge2edges': build_edge2edges(g.copy(),
                                                                targets)},
        truth=set([(i, j, (test_m[i, j]
                           if test_m[i, j] != 0
                           else test_m[j, i]))
                   for i, j in targets]))

    P_u = coo_matrix((data, (idx_i, idx_j)),
                     shape=test_m.shape).tocsr()
    print('dumping result...')
    pkl.dump(status, open('data/{}/status.pkl'.format(dataset), 'wb'))
    print('Prediction accuracy {}'.format(status.acc_list[-1]))
