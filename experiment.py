from pyspark.sql import SparkSession

from snpp.cores.joint_part_pred import iterative_approach
from snpp.cores.max_balance import greedy
from snpp.cores.lowrank import weighted_partition_sparse
from snpp.cores.budget_allocation import exponential_budget
from snpp.cores.louvain import best_partition_matrix
from snpp.utils.matrix import load_sparse_csr, \
    save_sparse_csr, \
    split_train_test, \
    difference_ratio_sparse, \
    delete_csr_entries
from snpp.utils.signed_graph import fill_diagonal, make_symmetric

dataset = 'slashdot'
method = 'lowrank'
lambda_ = 0.1
k = 40
max_iter = 20
random_seed = 123456

raw_mat_path = 'data/{}.npz'.format(dataset)


recache_input = True


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

        print(train_m.nnz)
        print(test_m.nnz)
        # some processing
        print('making symmetric...')
        train_m = make_symmetric(train_m)
        print('filling diagonal...')
        train_m = fill_diagonal(train_m)

        # remove overlap in train_m and test_m from test_m
        overlaps = set(zip(*train_m.nonzero())).intersection(set(zip(*test_m.nonzero())))
        print(len(overlaps), train_m.nnz, test_m.nnz)
        test_m = delete_csr_entries(test_m, overlaps)
        
        print('saving pre-split train and test matrix...')
        save_sparse_csr('data/slashdot/train_sym', train_m)
        save_sparse_csr('data/slashdot/test', test_m)
    else:
        print('loading pre-split train and test matrix...')
        train_m = load_sparse_csr('data/slashdot/train_sym.npz')
        test_m = load_sparse_csr('data/slashdot/test.npz')
    overlaps = set(zip(*train_m.nonzero())).intersection(set(zip(*test_m.nonzero())))
    assert len(overlaps) == 0, \
        'train and test has overlap {}: {}'.format(len(overlaps), overlaps)
    
    targets = list(zip(*test_m.nonzero()))
    print('#targets = {}'.format(len(targets)))
    A, P = iterative_approach(
        train_m, W=None, T=targets, k=k,
        # graph_partition_f=weighted_partition_sparse,
        # graph_partition_kwargs=dict(sc=spark_context,
        #                             lambda_=lambda_, iterations=max_iter,
        #                             seed=random_seed),
        graph_partition_f=best_partition_matrix,
        budget_allocation_f=exponential_budget,
        budget_allocation_kwargs=dict(exp_const=2),
        solve_maxbalance_f=greedy)

    error_rate = difference_ratio_sparse(test_m, P)
