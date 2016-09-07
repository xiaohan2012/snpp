import contexts as ctx

from snpp.utils.edge_filter import filter_by_min_triangle_count


def test_filter_by_min_triangle_count(g6):
    targets = [(0, 1), (2, 3), (4, 5)]
    assert list(filter_by_min_triangle_count(g6, targets, count=1)) == \
        targets

    assert list(filter_by_min_triangle_count(g6, targets, count=2)) == \
        [targets[0], targets[1]]

    assert list(filter_by_min_triangle_count(g6, targets, count=3)) == \
        [targets[0]]

    assert list(filter_by_min_triangle_count(g6, targets, count=4)) == []
