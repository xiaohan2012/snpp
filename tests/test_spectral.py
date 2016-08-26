import contexts as ctx
import pytest

from snpp.cores.kunegis2010 import build_L
from data import Q1, sparse_Q1

def test_sparse(Q1, sparse_Q1):
    build_kunegis2010(Q1)
