import numpy as np

from h2opt import folds


def test_group_folds_keep_groups_together_and_balance_folds():
    groups = np.repeat(np.arange(23), 3)
    fold = folds.group_folds(groups, n_folds=5, seed=1)
    for g in range(23):
        assert len(set(fold[groups == g])) == 1
    sizes = np.bincount(fold[::3], minlength=5)
    assert sizes.max() - sizes.min() <= 1

