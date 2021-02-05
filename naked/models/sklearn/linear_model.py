def linear_regression(x, coef_, intercept_):
    return intercept_ + sum(xi * coef_[i] for i, xi in enumerate(x))


def logistic_regression(x, coef_, intercept_):
    import math
    logits = intercept_[0] + sum(xi * coef_[0][i] for i, xi in enumerate(x))
    return 1 / (1 + math.exp(-logits))
