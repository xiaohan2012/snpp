"""
For the utilities
"""
import contexts as ctx
from snpp.utils.signed_graph import symmetric_stat
from data import Q1_d


def test_symmetric_stat(Q1_d):
    c_sym, c_consis = symmetric_stat(Q1_d)
    assert c_sym == 6
    assert c_consis == 4


