import importlib
import pickle

import naked
from naked import mapping
import streamlit as st

"""
# Convert a machine learning estimator to pure Python code

This is a tool that renders a pure Python representation of a pickled estimator. The output is
just a bunch of Python functions that don't require any dependencies whatsoever. This makes it
really trivial to put a machine learning model into production: you just have to copy/paste the
code into your application. The code generation is done with [`naked`](https://github.com/MaxHalford/naked).

The following estimators are supported:
"""

listing = ""

for name, estimators in mapping.items():
    lib = importlib.import_module(name)
    listing += f'* {name} {lib.__version__}\n'
    for estimator in estimators:
        listing += f'    * {estimator}\n'

st.markdown(listing)

uploaded_file = st.file_uploader("Choose a pickled model")

if uploaded_file:
    model = pickle.loads(uploaded_file.getvalue())
    st.code(naked.strip(model))
