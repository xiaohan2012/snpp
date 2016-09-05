import contexts as ctx
from snpp.cores.budget_allocation import max_budget, \
    linear_budget, \
    exponential_budget

default_args = {k: None
                for k in ('C', 'g')}


def test_max_budget():
    b = max_budget(iter_n=0, total_budget=10,
                   **default_args)
    assert b == 10


def test_linear_budget():
    b = linear_budget(iter_n=2, linear_const=10,
                      **default_args)
    assert b == 20


def test_exponential_budget():
    b = exponential_budget(iter_n=2, exp_const=10,
                           **default_args)
    assert b == 100
