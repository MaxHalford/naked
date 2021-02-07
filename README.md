# naked

`naked` allows you to strip a model and only keep what matters for making predictions. The result is a pure Python function with no third-party dependencies that you can simply copy/paste wherever you wish.

This is simpler than deploying an API endpoint or loading a serialized model. The jury is still out on whether this is sane or not.

## Installation

```sh
pip install git+https://github.com/MaxHalford/naked
```

## Examples

### `sklearn.linear_model.LinearRegression`

First, we fit a model.

```py
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
lin_reg = LinearRegression().fit(X, y)
lin_reg.fit(X, y)
```

Then, we strip it.

```py
import naked

print(naked.strip(lin_reg))
```

Which produces the following output.

```py
def linear_regression(x):

    coef_ = [1.0000000000000002, 1.9999999999999991]
    intercept_ = 3.0000000000000018

    return intercept_ + sum(xi * wi for xi, wi in enumerate(coef_))
```

### `sklearn.pipeline.Pipeline`

```py
import naked
from sklearn import linear_model
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import preprocessing

model = pipeline.make_pipeline(
    feature_extraction.text.TfidfVectorizer(),
    preprocessing.Normalizer(),
    linear_model.LogisticRegression(solver='liblinear')
)

docs = ['Sad', 'Angry', 'Happy', 'Joyful']
is_positive = [False, False, True, True]

model.fit(docs, is_positive)

print(naked.strip(model))
```

This produces the following output.

```py
def tfidf_vectorizer(x):

    lowercase = True
    norm = 'l2'
    vocabulary_ = {'sad': 3, 'angry': 0, 'happy': 1, 'joyful': 2}
    idf_ = [1.916290731874155, 1.916290731874155, 1.916290731874155, 1.916290731874155]

    import re

    if lowercase:
        x = x.lower()

    # Tokenize
    x = re.findall(r"(?u)\b\w\w+\b", x)
    x = [xi for xi in x if len(xi) > 1]

    # Count term frequencies
    from collections import Counter
    tf = Counter(x)
    total = sum(tf.values())

    # Compute the TF-IDF of each tokenized term
    tfidf = [0] * len(vocabulary_)
    for term, freq in tf.items():
        try:
            index = vocabulary_[term]
        except KeyError:
            continue
        tfidf[index] = freq * idf_[index] / total

    # Apply normalization
    if norm == 'l2':
        norm_val = sum(xi ** 2 for xi in tfidf) ** .5

    return [v / norm_val for v in tfidf]

def normalizer(x):

    norm = 'l2'

    if norm == 'l2':
        norm_val = sum(xi ** 2 for xi in x) ** .5
    elif norm == 'l1':
        norm_val = sum(abs(xi) for xi in x)
    elif norm == 'max':
        norm_val = max(abs(xi) for xi in x)

    return [xi / norm_val for xi in x]

def logistic_regression(x):

    coef_ = [[-0.40105811611957726, 0.40105811611957726, 0.40105811611957726, -0.40105811611957726]]
    intercept_ = [0.0]

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

def pipeline(x):
    x = tfidf_vectorizer(x)
    x = normalizer(x)
    x = logistic_regression(x)
    return x
```

## FAQ

### What models are supported?

```py
>>> import naked
>>> print(naked.AVAILABLE)
sklearn
    LinearRegression
    LogisticRegression
    Normalizer
    StandardScaler
    TfidfVectorizer

```

### Will this work for any version of <insert library name>?

Not by design. A release of `naked` is intended to support a library above a particular version. If we notice that `naked` doesn't work for a newer version of a given library, then a new version of `naked` should be released to handle said library version. You may refer to the [`pyproject.toml`](pyproject.toml) file to view library support.
## Development workflow

```sh
git clone https://github.com/MaxHalford/naked
cd naked
poetry install
poetry shell
pytest
```

## Things to do

- Implement more models. For instance it should quite straightforward to support LightGBM.
- Remove useless branching conditions. Parameters are currently handled via `if` statements. Ideally it would be nice to remove the `if` statements and only keep the code that will actually run.
## License

MIT
