def indexed_entries(sparse_matrix):
    """
    Args:
    
    Return:
    list of (row_id, col_id, value)
    """
    return [(i, j, sparse_matrix[i, j])
            for i, j in zip(*sparse_matrix.nonzero())]
