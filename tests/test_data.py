import contexts as ctx

import numpy as np
from snpp.utils.data import load_csv_network, example_for_intuition


def test_load_csv_network():
    g = load_csv_network(path=ctx.abs_path('data/epinions_truncated.txt'))
    assert g.number_of_nodes() == 67
    assert g.number_of_edges() == 66


def test_synthetic_data():
    n1, n2, p = 4, 5, 0.2
    Q, _ = example_for_intuition(n1, n2, p)
    nnz = np.count_nonzero(Q)
    assert (n1 * n2) ** 2 * p == (nnz - n1 * n2)
    assert (Q == np.transpose(Q)).all()