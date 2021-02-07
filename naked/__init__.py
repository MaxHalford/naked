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


def handle_input_names(x):
    return [x[name] for name in names]


def make_handle_input_names(names: str) -> Func:
    code = inspect.getsource(handle_input_names)
    loc = code.splitlines()
    return Func('handle_input_names', loc[0] + f'\n    names = {names}\n' + loc[1])


def handle_output_names(x):
    return dict(zip(names, x))


def make_handle_output_names(names: str) -> Func:
    code = inspect.getsource(handle_output_names)
    loc = code.splitlines()
    return Func('handle_output_names', loc[0] + f'\n    names = {names}\n' + loc[1])


def _strip(model):

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


def strip(model, input_names=None, output_names=None):

    if isinstance(model, Pipeline):
        func = FuncPipeline([_strip(step) for _, step in model.steps])
    else:
        func = _strip(model)

    # If input names are specified, then we'll assume that the input x is a dictionary and not a
    # list. We'll be able to handle by first mapping the dictionary values to a list in the
    # specified order.
    if input_names is not None:
        handle_input_names = make_handle_input_names(input_names)
        if isinstance(func, FuncPipeline):
            func.funcs.insert(0, handle_input_names)
        else:
            func = FuncPipeline([handle_input_names, func])

    # If output names are specified, then we'll produce a dictionary instead of a list.
    if output_names is not None:
        handle_output_names = make_handle_output_names(output_names)
        if isinstance(func, FuncPipeline):
            func.funcs.append(handle_output_names)
        else:
            func = FuncPipeline([func, handle_output_names])

    return func
