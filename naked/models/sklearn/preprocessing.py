def normalizer(x, norm):
    if norm == 'l2':
        norm_val = sum(xi ** 2 for xi in x) ** .5
    elif norm == 'l1':
        norm_val = sum(abs(xi) for xi in x)
    elif norm == 'max':
        norm_val = max(abs(xi) for xi in x)

    return [xi / norm_val for xi in x]
