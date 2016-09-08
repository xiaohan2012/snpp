from snpp.utils.signed_graph import g2m
from snpp.utils.data import load_train_test_graphs
from snpp.utils.edge_filter import filter_by_min_triangle_count

from snpp.cores.joint_part_pred import iterative_approach
from snpp.cores.max_balance import faster_greedy
from snpp.cores.lowrank import partition_graph
from snpp.cores.budget_allocation import constant_budget
from snpp.cores.triangle import build_edge2edges

from snpp.utils.spark import sc


dataset = 'slashdot'
lambda_ = 0.2
k = 10
max_iter = 100
random_seed = 123456

recache_input = False
min_tri_count = 20


train_g, test_g = load_train_test_graphs(dataset, recache_input)
train_g_ud = train_g.to_undirected()

confident_edges = set(filter_by_min_triangle_count(train_g_ud, test_g.edges(), min_tri_count))

part, iter_preds, status = iterative_approach(
        train_g_ud,
        T=confident_edges,
        k=k,
        graph_partition_f=partition_graph,
        graph_partition_kwargs=dict(sc=sc,
                                    lambda_=lambda_,
                                    iterations=max_iter,
                                    seed=random_seed),
        budget_allocation_f=constant_budget,
        budget_allocation_kwargs=dict(const=200),
        solve_maxbalance_f=faster_greedy,
        solve_maxbalance_kwargs={'edge2edges': build_edge2edges(train_g_ud.copy(),
                                                                confident_edges)},
        truth=set([(i, j, test_g[i][j]['sign'])
                   for i, j in confident_edges]),
        perform_last_partition=False
)
