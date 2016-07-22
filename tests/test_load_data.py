import contexts as ctx

from snpp.utils.data import load_csv_network


def test_load_csv_network():
    g = load_csv_network(path=ctx.abs_path('data/epinions_truncated.txt'))
    assert g.number_of_nodes() == 67
    assert g.number_of_edges() == 66
