from packaging.version import Version, parse

import platform
import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import torch

def setup_module():
    pass

def test_library_versions():
 
    # We use Python's distutils to verify versions. It seems sufficient for our purpose.
    # Alternatively, we could use pkg_resources.parse_version for a more robust parse.

    min_python = '3.6'
    min_numpy = '1.13'
    min_scipy = '1.0'
    min_torch = '1.0'
    min_pandas = '0.21'
    min_matplotlib = '2.0'

    assert parse(platform.python_version()) > parse(min_python)
    assert parse(np.__version__) > parse(min_numpy)
    assert parse(sp.__version__) > parse(min_scipy)
    assert parse(torch.__version__) > parse(min_torch)
    assert parse(pd.__version__) > parse(min_pandas)
    assert parse(matplotlib.__version__) > parse(min_matplotlib)
