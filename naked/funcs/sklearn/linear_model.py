def linear_regression(x, coef_, intercept_):
    return intercept_ + sum(xi * wi for xi, wi in enumerate(coef_))


def logistic_regression(x, coef_, intercept_):
    import math

    logits = [
        b + sum(xi * wi for xi, wi in zip(x, w))
        for w, b in zip(coef_, intercept_)
    ]

    # Sigmoid activation for binary classification
    if len(logits) == 1:
        p_true = 1 / (1 + math.exp(-logits[0]))
        return [1 - p_true, p_true]

    # Softmax activation for multi-class classification
    z_max = max(logits)
    exp = [math.exp(z - z_max) for z in logits]
    exp_sum = sum(exp)
    return [e / exp_sum for e in exp]
