def accuracy(test_g, preds):
    if not isinstance(preds, set):
        preds = set(preds)
    edges = [(i, j) for i, j, _ in preds]
    truth = set((i, j, test_g[i][j]['sign']) for i, j in edges)

    assert len(truth) == len(preds)
    for _, _, s in truth:
        assert s != 0
    assert set([(i, j) for i, j, _ in truth]) == set([(i, j) for i, j, _ in preds])

    return len(truth.intersection(preds)) / len(preds)
