import networkx as nx
from matrix import load_sparse_csr, indexed_entries


dataset = 'wiki'


def main():
    input_path = 'data/{}.npz'.format(dataset)
    output_path = 'output/{}-positive.gml'.format(dataset)
    m = load_sparse_csr(input_path)
    
    g = nx.DiGraph()
    g.add_edges_from((i, j)
                     for i, j, s in indexed_entries(m)
                     if s == 1)
    nx.write_gml(g, output_path, stringizer=str)
    
if __name__ == '__main__':
    main()
