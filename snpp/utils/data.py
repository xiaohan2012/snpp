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


def main():
    dataset = 'epinions'
    path = 'data/soc-sign-{}.txt'.format(dataset)
    g = load_csv_network(path)
    nx.write_gpickle(g, 'data/{}.pkl'.format(dataset))


if __name__ == '__main__':
    main()
