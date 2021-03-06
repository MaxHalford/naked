import importlib
import pickle

import naked
from naked import mapping
import streamlit as st

"""
# Convert a model to pure Python

This is an interface to display a pure Python representation of a pickled model. It uses
[`naked`](https://github.com/MaxHalford/naked) under the hood.

The following library versions are supported:
"""

for name in mapping:
    lib = importlib.import_module(name)
    st.write(f'- {name} {lib.__version__}')

uploaded_file = st.file_uploader("Choose a pickled model")

if uploaded_file:
    model = pickle.loads(uploaded_file.getvalue())
    st.code(naked.strip(model))
