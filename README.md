<h1>naked</h1>

`naked` is a Python tool which allows you to strip a model and only keep what matters for making predictions. The result is a pure Python function with no third-party dependencies that you can simply copy/paste wherever you wish.

This is simpler than deploying an API endpoint or loading a serialized model. The jury is still out on whether this is sane or not. Of course I'm not the first one to have done this, for instance see [sklearn-porter](https://github.com/nok/sklearn-porter) and [pure-predict](https://github.com/Ibotta/pure-predict). Note if you don't mind installing depencencies on your inference machine, then tools such as [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) might do the job for you. It's also worth mentioning [xgb2sql](https://github.com/Chryzanthemum/xgb2sql), which converts an XGBoost model to a SQL query.

Note that you can use `naked` via this [web interface](https://naked-app.herokuapp.com/).

- [Installation](#installation)
- [Examples](#examples)
  - [`sklearn.linear_model.LinearRegression`](#sklearnlinear_modellinearregression)
  - [`sklearn.pipeline.Pipeline`](#sklearnpipelinepipeline)
- [FAQ](#faq)
  - [What models are supported?](#what-models-are-supported)
  - [Will this work for all library versions?](#will-this-work-for-all-library-versions)
  - [How can I trust this is correct?](#how-can-i-trust-this-is-correct)
  - [How should I handle feature names?](#how-should-i-handle-feature-names)
  - [What about output names?](#what-about-output-names)
- [Development workflow](#development-workflow)
- [Things to do](#things-to-do)
- [License](#license)

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

### Will this work for all library versions?

Not by design. A release of `naked` is intended to support a library above a particular version. If we notice that `naked` doesn't work for a newer version of a given library, then a new version of `naked` should be released to handle said library version. You may refer to the [`pyproject.toml`](pyproject.toml) file to view library support.

### How can I trust this is correct?

This package is really easy to unit test. One simply has to compare the outputs of the model with its "naked" version and check that the outputs are identical. Check out the [`test_naked.py`](naked/test_naked.py) file if you're curious.

### How should I handle feature names?

Let's take the example of a multi-class logistic regression trained on the wine dataset.

```py
from sklearn import datasets
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing

dataset = datasets.load_wine()
X = dataset.data
y = dataset.target
model = pipeline.make_pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)
model.fit(X, y)
```

By default, the `strip` function produces a function that takes as input a list of feature values. Instead, let's say we want to evaluate the function on a dictionary of features, thus associating each feature value with a name.

```py
x = dict(zip(dataset.feature_names, X[0]))
print(x)
```

```py
{'alcohol': 14.23,
 'malic_acid': 1.71,
 'ash': 2.43,
 'alcalinity_of_ash': 15.6,
 'magnesium': 127.0,
 'total_phenols': 2.8,
 'flavanoids': 3.06,
 'nonflavanoid_phenols': 0.28,
 'proanthocyanins': 2.29,
 'color_intensity': 5.64,
 'hue': 1.04,
 'od280/od315_of_diluted_wines': 3.92,
 'proline': 1065.0}
```

Passing the feature names to the `strip` function will add a function that maps the features to a list.

```py
naked.strip(model, input_names=dataset.feature_names)
```

```py
def handle_input_names(x):
    names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    return [x[name] for name in names]

def standard_scaler(x):

    mean_ = [13.000617977528083, 2.336348314606741, 2.3665168539325854, 19.49494382022472, 99.74157303370787, 2.295112359550562, 2.0292696629213474, 0.36185393258426973, 1.5908988764044953, 5.058089882022473, 0.9574494382022468, 2.6116853932584254, 746.8932584269663]
    var_ = [0.6553597304633259, 1.241004080924126, 0.07484180027774268, 11.090030614821362, 202.84332786264366, 0.3894890323191514, 0.9921135115515715, 0.015401619113748266, 0.32575424820098453, 5.344255847629093, 0.05195144969069561, 0.5012544628203511, 98609.60096578706]
    with_mean = True
    with_std = True

    def scale(x, m, v):
        if with_mean:
            x -= m
        if with_std:
            x /= v ** .5
        return x

    return [scale(xi, m, v) for xi, m, v in zip(x, mean_, var_)]

def logistic_regression(x):

    coef_ = [[0.8101347947338147, 0.20382073148760085, 0.47221241678911957, -0.8447843882542064, 0.04952904623674445, 0.21372479616642068, 0.6478750705319883, -0.19982499112990385, 0.13833867563545404, 0.17160966151451867, 0.13090887117218597, 0.7259506896985365, 1.07895948707047], [-1.0103233753629153, -0.44045952703036084, -0.8480739967718842, 0.5835732316278703, -0.09770602368275362, 0.027527982220605866, 0.35399157401383297, 0.21278279386396404, 0.2633610495737497, -1.0412707677956505, 0.6825215991118386, 0.05287634940648419, -1.1407929345327175], [0.20018858062910203, 0.23663879554275832, 0.37586157998276365, 0.26121115662633365, 0.048176977446007865, -0.2412527783870254, -1.0018666445458222, -0.012957802734061021, -0.40169972520920566, 0.8696611062811332, -0.8134304702840255, -0.7788270391050198, 0.061833447462247046]]
    intercept_ = [0.41229358315867787, 0.7048164121833935, -1.1171099953420585]

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
    x = handle_input_names(x)
    x = standard_scaler(x)
    x = logistic_regression(x)
    return x
```

### What about output names?

You can also specify the `output_names` parameter to associate each output value with a name. Of course, this doesn't work for cases where a single value is produced, such as single-target regression.


```py
naked.strip(model, input_names=dataset.feature_names, output_names=dataset.target_names)
```

```py
def handle_input_names(x):
    names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    return [x[name] for name in names]

def standard_scaler(x):

    mean_ = [13.000617977528083, 2.336348314606741, 2.3665168539325854, 19.49494382022472, 99.74157303370787, 2.295112359550562, 2.0292696629213474, 0.36185393258426973, 1.5908988764044953, 5.058089882022473, 0.9574494382022468, 2.6116853932584254, 746.8932584269663]
    var_ = [0.6553597304633259, 1.241004080924126, 0.07484180027774268, 11.090030614821362, 202.84332786264366, 0.3894890323191514, 0.9921135115515715, 0.015401619113748266, 0.32575424820098453, 5.344255847629093, 0.05195144969069561, 0.5012544628203511, 98609.60096578706]
    with_mean = True
    with_std = True

    def scale(x, m, v):
        if with_mean:
            x -= m
        if with_std:
            x /= v ** .5
        return x

    return [scale(xi, m, v) for xi, m, v in zip(x, mean_, var_)]

def logistic_regression(x):

    coef_ = [[0.8101347947338147, 0.20382073148760085, 0.47221241678911957, -0.8447843882542064, 0.04952904623674445, 0.21372479616642068, 0.6478750705319883, -0.19982499112990385, 0.13833867563545404, 0.17160966151451867, 0.13090887117218597, 0.7259506896985365, 1.07895948707047], [-1.0103233753629153, -0.44045952703036084, -0.8480739967718842, 0.5835732316278703, -0.09770602368275362, 0.027527982220605866, 0.35399157401383297, 0.21278279386396404, 0.2633610495737497, -1.0412707677956505, 0.6825215991118386, 0.05287634940648419, -1.1407929345327175], [0.20018858062910203, 0.23663879554275832, 0.37586157998276365, 0.26121115662633365, 0.048176977446007865, -0.2412527783870254, -1.0018666445458222, -0.012957802734061021, -0.40169972520920566, 0.8696611062811332, -0.8134304702840255, -0.7788270391050198, 0.061833447462247046]]
    intercept_ = [0.41229358315867787, 0.7048164121833935, -1.1171099953420585]

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

def handle_output_names(x):
    names = ['class_0' 'class_1' 'class_2']
    return dict(zip(names, x))

def pipeline(x):
    x = handle_input_names(x)
    x = standard_scaler(x)
    x = logistic_regression(x)
    x = handle_output_names(x)
    return x
```

As you can see, by specifying `input_names` as well as `output_names`, we obtain a pipeline of functions which takes as input a dictionary and produces a dictionary.
## Development workflow

```sh
git clone https://github.com/MaxHalford/naked
cd naked
poetry install
poetry shell
pytest
```

You may test the web interface locally by running streamlit:

```sh
streamlit run app/app.py
```

## Things to do

- Implement more models. For instance it should quite straightforward to support LightGBM.
- Remove useless branching conditions. Parameters are currently handled via `if` statements. Ideally it would be nice to remove the `if` statements and only keep the code that will actually run. This should be doable by using the `ast` module.
## License

MIT
