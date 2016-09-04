from .triangle import first_order_triangles_count
from scipy.sparse import dok_matrix
from copy import copy


def edge_weight_sum(edges, W):
    if W is None:
        return len(edges)
    else:
        return sum(W[n1, n2] for n1, n2 in edges)


def greedy(A, W, C, B, T):
    """
    Args:
    
    A: sign matrix (lil_matrix or csr_matrix)
    W: edge weight matrix
       or None (means \forall (i, j), W[i, j] = 1)
    C: cluster label array
    B: budget
    T: set of target edges (node i, node j)

    Returns:
    P: predicted sign matrix (csr)
    """
    P = dok_matrix(A.shape)

    assert isinstance(T, set)
    targets = copy(T)

    T_p = set()
    while (edge_weight_sum(T_p, W) < B
           and len(targets) > 0):
        try:
            n1, n2, s, c = max(first_order_triangles_count(A, C, targets),
                               key=lambda tpl: tpl[-1])
        except ValueError:  # no first-order triangles
            print("WARN: empty first-order triangles. So exit loop")
            print('targets:')
            print(targets)
            break
        
        print('assigning {} to ({}, {}) produces {} more balanced triangles'.format(
            s, n1, n2, c
        ))
        
        T_p.add((n1, n2))
        targets.remove((n1, n2))
        P[n1, n2] = s
    return P.tocsr()
