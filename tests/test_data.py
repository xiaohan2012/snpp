import contexts as ctx

import numpy as np
from snpp.utils.data import load_csv_network, \
    example_for_intuition, \
    load_csv_as_sparse, \
    parse_wiki_rfa


def test_load_csv_network():
    g = load_csv_network(path=ctx.abs_path('data/epinions_truncated.txt'))
    assert g.number_of_nodes() == 67
    assert g.number_of_edges() == 66


def test_load_csv_as_sparse():
    m = load_csv_as_sparse(path=ctx.abs_path('data/signed_graph_4_nodes.csv'))
    assert m.shape == (4, 4)
    assert m.nnz == 4
    assert m[0, 1] == 1
    assert m[1, 2] == -1
    

def test_synthetic_data():
    n1, n2, p = 4, 5, 0.2
    N = n1 * n2
    Q, _ = example_for_intuition(n1, n2, p)
    nnz = np.count_nonzero(Q)
    
    assert N * (N - 1) / 2 * p == (nnz - N) / 2
    assert (Q == np.transpose(Q)).all()


def test_parse_wiki_rfa():
    df = parse_wiki_rfa('data/wiki_rfa.txt')
    assert df.shape == (2, 7)

    assert df.iloc[0]['SRC'] == 'Pakaran'
    assert df.iloc[0]['TGT'] == 'WhisperToMe'
    assert df.iloc[0]['VOT'] == 1

    assert df.iloc[1]['SRC'] == 'Jimregan'
    assert df.iloc[1]['TGT'] == 'Zanimum'
    assert df.iloc[1]['VOT'] == -1
