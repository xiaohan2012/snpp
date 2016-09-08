import contexts as ctx
from snpp.cores.budget_allocation import max_budget, \
    linear_budget, \
    exponential_budget, \
    constant_then_exponential_budget, \
    constant_budget

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


def test_constant_then_exponential_budget():
    for i in range(1, 6):
        b = constant_then_exponential_budget(iter_n=i, const=50, exp_const=2, switch_iter=6, **default_args)
        assert b == 50
    b = constant_then_exponential_budget(iter_n=6, const=50, exp_const=2, switch_iter=6, **default_args)
    assert b == 64
    b = constant_then_exponential_budget(iter_n=7, const=50, exp_const=2, switch_iter=6, **default_args)
    assert b == 128


def test_constant_const_budget():
    assert constant_budget(None, None, None, const=100, edge2true_sign=None) == 100

