import findspark  # this needs to be the first import
findspark.init()

import networkx as nx
from pyspark import SparkConf
from pyspark import SparkContext
from snpp.cores.lowrank import partition_graph


conf = (SparkConf().setMaster("local[2]").setAppName("SparkTest"))
sc = SparkContext(conf=conf)
sc.setCheckpointDir('checkpoint')  # Stackoverflow error

train_graph_path = 'data/{}/train_graph.pkl'.format('slashdot')
g = nx.read_gpickle(train_graph_path)
partition_graph(g, k=40, sc=sc,
                lambda_=0.1,
                iterations=20,
                seed=123456)

sc.stop()
