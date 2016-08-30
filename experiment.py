from pyspark.sql import SparkSession

from snpp.cores.joint_part_pred import iterative_approach, \
    naive_approach
from snpp.cores.max_balance import greedy
from snpp.cores.lowrank import weighted_partition_sparse
from snpp.cores.budget_allocation import exponential_budget
from snpp.utils.matrix import load_sparse_csr, split_train_dev_test
from snpp.utils.signed_graph import fill_diagonal, make_symmetric

dataset = 'slashdot'
method = 'lowrank'
lambda_ = 0.1
k = 10
max_iter = 20
random_seed = 123456

raw_mat_path = 'data/{}.npz'.format(dataset)


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Signed Network Experiment")\
        .getOrCreate()
    spark_context = spark.sparkContext

    m = load_sparse_csr(raw_mat_path)

    train_m, dev_m, test_m = split_train_dev_test(
        m,
        weights=[0.8, 0.1, 0.1])

    # some processing
    print('making symmetric...')
    train_m = make_symmetric(train_m)
    print('filling diagonal...')
    train_m = fill_diagonal(train_m)
    
    targets = zip(*test_m.nonzero())
    A, P = iterative_approach(
        train_m, W=None, T=targets,
        graph_partition_f=weighted_partition_sparse,
        graph_partition_kwargs=dict(sc=spark_context,
                                    lambda_=lambda_, iterations=max_iter,
                                    seed=random_seed),
        budget_allocation_f=exponential_budget,
        budget_allocation_kwargs=dict(exp_const=2),
        solve_maxbalance_f=greedy)

    
