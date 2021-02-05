# naked

🚧 Work in progress!

Strip a model and only keep what matters for prediction. This way you can just copy/paste your model in production.

## Examples

### scikit-learn's `LinearRegression`

First we fit a model:

```py
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
lin_reg = LinearRegression().fit(X, y)
lin_reg.fit(X, y)
```

Then we strip it:

```py
import naked

naked.strip(lin_reg)
```

This prints out:

```
def linear_regression(x):

    coef_ = [1.0000000000000002, 1.9999999999999991]
    intercept_ = 3.0000000000000018

    return intercept_ + sum(xi * coef_[i] for i, xi in enumerate(x))
```

### scikit-learn's `Pipeline`

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

naked.strip(model)
```

```py
def tfidf_vectorizer(x):

    norm = 'l2'
    vocabulary_ = {'sad': 3, 'angry': 0, 'happy': 1, 'joyful': 2}
    idf_ = [1.916290731874155, 1.916290731874155, 1.916290731874155, 1.916290731874155]


    import re

    x = x.lower()
    x = re.findall(r"(?u)\b\w\w+\b", x)
    x = [xi for xi in x if len(xi) > 1]

    from collections import Counter
    tf = Counter(x)
    total = sum(tf.values())

    tfidf = [0] * len(vocabulary_)

    for term, freq in tf.items():
        try:
            index = vocabulary_[term]
        except KeyError:
            continue
        tfidf[index] = freq * idf_[index] / total

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

def linear_regression(x):

    coef_ = [[-0.40105811611957726, 0.40105811611957726, 0.40105811611957726, -0.40105811611957726]]
    intercept_ = [0.0]

    return intercept_ + sum(xi * coef_[i] for i, xi in enumerate(x))

def pipeline(x):
    x = tfidf_vectorizer(x)
    x = normalizer(x)
    x = linear_regression(x)
    return x
```
