import networkx as nx
from data import load_csv_network


def positive_subgraph(g):
    return nx.Graph((s, t) for s, t in g.edges_iter() if g[s][t]['sign'] == 1)


def main():
    input_path = 'data/soc-sign-epinions.txt'
    output_path = 'output/epinions-positive.gml'
    g = load_csv_network(input_path)
    g = positive_subgraph(g)

    nx.write_gml(g, output_path, stringizer=str)

    
if __name__ == '__main__':
    main()
