import inspect
import re
import types

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import funcs


__all__ = ['strip', 'AVAILABLE_MODELS']

class Func:

    def __init__(self, name, code):
        self.name = name
        self.code = code
        code = compile(code, "<string>", "exec")
        self.func = types.FunctionType(code.co_consts[0], globals(), name)

    def __call__(self, x):
        return self.func(x)

    def __repr__(self):
        return self.code.replace('\n\n\n', '\n\n')

class FuncPipeline:

    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for func in self.funcs:
            x = func(x)
        return x

    def __repr__(self):
        code = '\n\n'.join(repr(func) for func in self.funcs)

        code += '\n\n'
        code += 'def pipeline(x):\n'
        for func in self.funcs:
            code += f'    x = {func.name}(x)\n'
        code += '    return x'

        return code


mapping = {
    'sklearn': {
        'LinearRegression': funcs.sklearn.linear_model.linear_regression,
        'LogisticRegression': funcs.sklearn.linear_model.logistic_regression,
        'Normalizer': funcs.sklearn.preprocessing.normalizer,
        'StandardScaler': funcs.sklearn.preprocessing.standard_scaler,
        'TfidfVectorizer': funcs.sklearn.feature_extraction.text.tfidf_vectorizer
    }
}


AVAILABLE = '\n'.join(
    mod + '\n' + '\n'.join(f'    {m}' for m in sorted(mapping[mod]))
    for mod in mapping
)

def strip(model):

    if isinstance(model, Pipeline):
        return FuncPipeline([strip(step) for _, step in model.steps])

    # Check if the model is supported
    mod = model.__class__.__module__.split('.')[0]
    try:
        func = mapping[mod][model.__class__.__name__]
    except KeyError:
        raise KeyError(f"I don't know how to unstrip {model.__class__.__name__} from {mod}.")

    # The model needs to have called fit
    check_is_fitted(model)

    # Now we just have to edit the function's source code by inserting the parameters
    code = inspect.getsource(func)
    code = re.sub(r'\(.+\)', '(x)', code, count=1)

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
