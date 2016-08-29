

def max_budget(C, A, P, iter_n,
               total_budget=None):
    """
    Args:
    
    C: cluster label array
    A: input sign matrix
    P: predicted sign matrix
    iter_n: iteration count
    total_budget: maximum budget we can have

    Returns:
    
    budget: number
    """
    return total_budget
    

def linear_budget(C, A, P, iter_n, linear_const=1):
    """
    budget = iter_n x constant
    """
    assert iter_n > 0
    return iter_n * linear_const


def exponential_budget(C, A, P, iter_n, exp_const=2):
    """
    budget = exp_const ^ iter_n
    """
    assert iter_n > 0
    return exp_const ** iter_n
    

