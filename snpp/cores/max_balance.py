import networkx as nx

from scipy.sparse import dok_matrix
from copy import copy

from .triangle import first_order_triangles_count, \
    first_order_triangles_count_g, \
    first_order_triangles_net_count_g, \
    build_edge2edges


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

# DEPRECATED
def greedy_g(g, C, B, T):
    """
    Args:
    
    g: signed network
    C: cluster label array
    B: budget
    T: set of target edges (node i, node j)

    Returns:
    predictions: (i, j, sign)
    """
    assert isinstance(g, nx.Graph)
    g = g.copy()
    
    assert isinstance(T, set)
    targets = copy(T)

    preds = []
    
    T_p = set()
    
    while True:
        budget_used = sum(g[i][j]['weight'] for i, j in T_p)
        if (budget_used >= B or len(targets) <= 0):
            break
        try:
            n1, n2, s, nc, ck = max(first_order_triangles_net_count_g(g, C, targets),
                                    key=lambda tpl: tpl[3])
        except ValueError:  # no first-order triangles
            print("WARN: empty first-order triangles. So exit loop")
            print('targets: {}'.format(targets))
            break
        
        print('assigning {} to ({}, {}) produces {} more balanced triangles {}'.format(
            s, n1, n2, nc, ck
        ))
        
        T_p.add((n1, n2))
        targets.remove((n1, n2))

        g.add_edge(n1, n2, weight=1, sign=s)
        preds.append((n1, n2, s))
    return preds


def faster_greedy(g, C, B, T, edge2edges=None):
    """
    Faster version that computes the triangle count only when necessary

    Args:
    
    g: signed network
    C: cluster label array
    B: budget
    T: set of target edges (node i, node j)
    edge2edges: dict of edge to set edges, each of which are in some triangle with the key edge

    Returns:
    predictions: (i, j, sign)
    """
    assert isinstance(g, nx.Graph)
    
    assert isinstance(T, set)
    targets = copy(T)

    preds = []
    
    T_p = set()

    print('building triangle_count_by_edge')
    triangle_count_by_edge = {}  # needs to be updated on the fly
    for n1, n2, s, sc, ck, info in first_order_triangles_net_count_g(g, C, targets):
        triangle_count_by_edge[(n1, n2)] = (s, sc, ck, info)

    if edge2edges is None:
        g = g.copy()
        edge2edges = build_edge2edges(g, T)
    else:
        print('edge2edges (size {}) is given'.format(len(edge2edges)))
        assert isinstance(edge2edges, dict)

    while True:
        budget_used = sum(g[i][j]['weight'] for i, j in T_p)
        if (budget_used >= B or len(targets) <= 0):
            break

        # find best edge
        # can use a heap here
        best_e = max(targets,
                     key=lambda k: triangle_count_by_edge[k][1])
        best_s, nc, ck, info = triangle_count_by_edge[best_e]

        print('assigning {} to {} produces {} more balanced triangles {}: {}'.format(
            best_s, best_e, nc, ck, info
        ))

        # update triangle information on affected edges
        # only consider un-predicted edges
        affected_edges = list(filter(lambda e: e not in T_p,
                                     edge2edges[(n1, n2)]))
        
        for i, j, s, nc, ck, info in first_order_triangles_net_count_g(g, C, affected_edges):
            triangle_count_by_edge[(i, j)] = (s, nc, ck, info)

        # gather result
        T_p.add(best_e)
        targets.remove(best_e)

        n1, n2 = best_e
        g.add_edge(n1, n2, weight=1, sign=best_s)
        preds.append((n1, n2, best_s))
    return preds
