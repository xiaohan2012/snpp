import contexts as ctx

import numpy as np
from nose.tools import assert_raises

from snpp.cores.triangle import extract_nodes_and_signs, \
    in_different_partitions, \
    can_be_balanced, \
    get_sign_1st_order, \
    first_order_triangles_count


def test_extract_nodes_and_signs():
    # correct case
    e = (0, 1)
    e1 = (1, 2, 1)
    e2 = (0, 2, -1)
    nodes, signs = extract_nodes_and_signs(e, e1, e2)
    assert nodes == {0, 1, 2}
    assert signs == [1, -1]

    # invalid case
    e = (0, 3)
    e1 = (1, 2, 1)
    e2 = (0, 2, -1)
    assert_raises(AssertionError, extract_nodes_and_signs, e, e1, e2)


def test_in_different_partitions():
    C1 = np.array(['a', 'b', 'c'])
    assert in_different_partitions([0, 1, 2], C1)

    C2 = np.array(['a', 'a', 'b'])
    assert not in_different_partitions([0, 1, 2], C2)


def test_can_be_balanced():
    e, e1, e2 = (0, 1), (1, 2, 1), (0, 2, -1)
    C1 = np.array(['a', 'b', 'c'])
    assert not can_be_balanced(e, e1, e2, C1)

    C2 = np.array(['a', 'b', 'b'])
    assert can_be_balanced(e, e1, e2, C2)


def test_get_sign_1st_order():
    e, e1, e2 = (0, 1), (1, 2, 1), (0, 2, -1)
    C1 = np.array(['a', 'b', 'c'])

    assert_raises(AssertionError, get_sign_1st_order, e, e1, e2, C1)
