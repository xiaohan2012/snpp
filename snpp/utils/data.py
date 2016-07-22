import pandas as pd
import networkx as nx


def load_csv_network(path):
    df = pd.read_csv(path, sep='\t')
    g = nx.DiGraph()
    g.name = path

    g.add_edges_from(
        (r['FromNodeId'], r['ToNodeId'], {'sign': r['Sign']})
        for i, r in df.iterrows()
    )
    return g
