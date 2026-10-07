import numpy as np

from h2opt import folds


def test_group_folds_keep_groups_together_and_balance_folds():
    groups = np.repeat(np.arange(23), 3)
    fold = folds.group_folds(groups, n_folds=5, seed=1)
    for g in range(23):
        assert len(set(fold[groups == g])) == 1
    sizes = np.bincount(fold[::3], minlength=5)
    assert sizes.max() - sizes.min() <= 1


def test_cross_validate_averages_fold_scores():
    groups = np.repeat(np.arange(10), 2)
    X = np.arange(20.0)

    def score_fold(fit, val):
        assert not set(groups[fit]) & set(groups[val])
        return [val.sum(), X[val].sum()]

    scores, fold_scores = folds.cross_validate(groups, score_fold, n_folds=5)
    assert fold_scores.shape == (5, 2)
    assert fold_scores[:, 0].sum() == 20 and fold_scores[:, 1].sum() == X.sum()
    np.testing.assert_allclose(scores, fold_scores.mean(axis=0))
