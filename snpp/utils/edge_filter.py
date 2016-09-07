
def filter_by_min_triangle_count(g, targets, count):
    g = g.copy()
    g.add_edges_from(targets)
    return filter(lambda e:
                  len(set(g[e[0]]).intersection(set(g[e[1]]))) >= count,
                  targets)

        
