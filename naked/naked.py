import inspect
import re
import types

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import models


class Func:

    def __init__(self, name, code):
        self.name = name
        self.code = code
        code = compile(code, "<string>", "exec")
        self.func = types.FunctionType(code.co_consts[0], globals(), name)

    def __call__(self, x):
        return self.func(x)

    def __repr__(self):
        return self.code

class FuncPipeline:

    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for func in self.funcs:
            x = func(x)
        return x

    def __repr__(self):
        code = '\n\n'.join(func.code for func in self.funcs)

        code += '\n\n'
        code += 'def pipeline(x):\n'
        for func in self.funcs:
            code += f'    x = {func.name}(x)\n'
        code += '    return x'

        return code


mapping = {
    ('sklearn', 'LinearRegression'): models.sklearn.linear_model.linear_regression,
    ('sklearn', 'LogisticRegression'): models.sklearn.linear_model.linear_regression,
    ('sklearn', 'Normalizer'): models.sklearn.preprocessing.normalizer,
    ('sklearn', 'TfidfVectorizer'): models.sklearn.feature_extraction.text.tfidf_vectorizer,
}


def strip(model):

    if isinstance(model, Pipeline):
        return FuncPipeline([undress(step) for _, step in model.steps])

    check_is_fitted(model)

    func = mapping[model.__class__.__module__.split('.')[0], model.__class__.__name__]
    code = inspect.getsource(func)
    code = re.sub('\(.+\)', '(x)', code, count=1)

    params_code = ''

    for param_name in inspect.signature(func).parameters:
        if param_name == 'x':
            continue

        param_val = getattr(model, param_name)
        if isinstance(param_val, np.ndarray):
            param_val = param_val.tolist()
        if isinstance(param_val, str):
            param_val = f"'{param_val}'"

        params_code += f'    {param_name} = {param_val}\n'

    # Insert the parameter specification code
    loc = code.splitlines()
    code = loc[0] + '\n\n' + params_code + '\n' + '\n'.join(loc[1:])

    return Func(func.__name__, code)
