"""The sorghum example data in data/examples, shared by the tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

import h2opt

# The tests are small: a few threads are enough, and the suite must not take every core of a
# machine that is running other jobs.
torch.set_num_threads(4)

EXAMPLES = Path(__file__).resolve().parent.parent / 'data' / 'examples'


@pytest.fixture(scope='session')
def examples():
    """Directory of the example data."""
    return EXAMPLES


@pytest.fixture(scope='session')
def sorghum():
    """Measurements X (plants x wavelengths), genotype labels and environment of the sorghum
    examples."""
    X = np.concatenate([h2opt.load_npz(EXAMPLES / f'X_file{i}.npz') for i in (1, 2)])
    groups = h2opt.load_npz(EXAMPLES / 'genotypes.npz')
    return X, groups, h2opt.load_npz(EXAMPLES / 'environment.npz')


@pytest.fixture(scope='session')
def sorghum_small(sorghum):
    """The sorghum examples on their first 200 genotype labels (460 plants), for the slow
    estimators: a Henderson3 step costs time in proportion to the individuals."""
    X, groups, environment = sorghum
    keep = np.isin(groups, np.unique(groups)[:200])
    return X[keep], groups[keep], environment[keep]
