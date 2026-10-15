"""Loading data files."""

import numpy as np


def load_npz(path, allow_pickle=False):
    """Load the single unnamed array (arr_0) stored in a .npz file."""
    return np.load(path, allow_pickle=allow_pickle)['arr_0']
