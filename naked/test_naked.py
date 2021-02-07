import math

import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import naked


@pytest.mark.parametrize("model", [
    LinearRegression()
])
@pytest.mark.parametrize("dataset", [
    datasets.make_regression(),
    datasets.load_boston(return_X_y=True),
    datasets.load_diabetes(return_X_y=True)
])
def test_reg(model, dataset):

    X, y = dataset
    model.fit(X, y)
    func = naked.strip(model)

    for x, yp in zip(X, model.predict(X)):
        assert math.isclose(func(x), yp)


@pytest.mark.parametrize("model", [
    make_pipeline(StandardScaler(), LogisticRegression())
])
@pytest.mark.parametrize("dataset", [
    # Binary
    datasets.make_classification(),
    datasets.load_breast_cancer(return_X_y=True),
    # Multi-class
    datasets.load_wine(return_X_y=True)
])
def test_clf(model, dataset):

    X, y = dataset
    model.fit(X, y)
    func = naked.strip(model)

    for x, yp in zip(X, model.predict_proba(X)):
        assert np.allclose(func(x), yp)
