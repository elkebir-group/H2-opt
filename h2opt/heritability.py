"""Heritability (ANOVA, Henderson's Method III) and genetic covariance of traits, differentiable in
PyTorch."""

import numpy as np
import torch


def _as_environment(environment, n):
    """Environment as an (n, g) array of categorical labels; None means no environment."""
    if environment is None:
        return np.zeros((n, 0))
    environment = np.asarray(environment)
    if environment.ndim == 1:
        environment = environment.reshape((-1, 1))
    if environment.shape[0] == 0:
        return np.zeros((n, 0))
    return environment


def _codes(labels, device):
    """Labels as integer codes 0..c-1 (a tensor on device) and the number c of labels."""
    _, inverse = np.unique(np.asarray(labels), return_inverse=True)
    inverse = inverse.reshape(-1)
    return torch.tensor(inverse, device=device), int(inverse.max()) + 1 if len(inverse) else 0


def _segment_sum(Y, codes, n_codes):
    """(n_codes, k) sums of the rows of the (n, k) tensor Y by their code."""
    return torch.zeros((n_codes, Y.shape[1]), dtype=Y.dtype, device=Y.device).index_add(
        0, codes, Y)


def _center_by(Y, mask, codes, n_codes):
    """Y minus the mean of its column over the rows of each code, among the rows where mask is 1;
    0 where mask is 0."""
    means = _segment_sum(Y * mask, codes, n_codes) / _segment_sum(mask, codes, n_codes).clamp(min=1)
    return (Y - means.index_select(0, codes)) * mask


class AnovaDesign:
    """Groups and environment of fixed individuals, preprocessed once for repeated calls.

    AnovaDesign(groups, environment, device).heritability(Y) equals
    anova_heritability(Y, groups, environment) but encodes the labels only once (e.g. once per
    training run instead of every step). With rows, an (n, k) boolean array, column j of Y is
    a trait of the individuals where column j of rows is true (the other entries of Y are ignored),
    so traits on different subsets of the individuals are computed together.
    """

    def __init__(self, groups, environment=None, device='cpu', rows=None):
        groups = np.asarray(groups)
        environment = _as_environment(environment, len(groups))
        self.groups, self.n_groups = _codes(groups, device)
        self.environment = [_codes(environment[:, a], device) for a in range(environment.shape[1])]
        rows = np.ones((len(groups), 1), dtype=bool) if rows is None else np.asarray(rows, bool)
        mask = torch.tensor(rows, dtype=torch.float64, device=device)
        # groups with one individual (among the rows of a column) are dropped
        counts = _segment_sum(mask, self.groups, self.n_groups)
        self._keep = mask * (counts[self.groups] >= 2)
        self._keep_float = {}

    def keep(self, dtype):
        """(n, k or 1) mask of the individuals that count, in the given dtype (cached)."""
        if dtype not in self._keep_float:
            self._keep_float[dtype] = self._keep.to(dtype)
        return self._keep_float[dtype]

    def heritability(self, Y, return_variance=False, env_adjusted_total=False):
        """See anova_heritability."""
        mask = self.keep(Y.dtype)
        n = mask.sum(axis=0)
        Y2 = (Y - (Y * mask).sum(axis=0) / n) * mask
        if not return_variance:
            Y2 = Y2 / ((torch.abs(Y2) * mask).sum(axis=0) / n)
        variance_total = torch.sum(Y2 ** 2, axis=0)

        if len(self.environment) == 0:
            variance_env = variance_total
        else:
            for codes, n_codes in self.environment:
                Y2 = _center_by(Y2, mask, codes, n_codes)
            Y2 = (Y2 - Y2.sum(axis=0) / n) * mask
            variance_env = torch.sum(Y2 ** 2, axis=0)

        sums = _segment_sum(Y2, self.groups, self.n_groups)
        sums_sq = _segment_sum(Y2 ** 2, self.groups, self.n_groups)
        sizes = _segment_sum(mask, self.groups, self.n_groups)
        within = (sums_sq - sums ** 2 / sizes.clamp(min=1)) * (sizes / (sizes - 1).clamp(min=1))
        variance_within = torch.sum(within, axis=0)

        if env_adjusted_total:
            variance_total = variance_env

        if return_variance:
            return (variance_env - variance_within) / n, variance_total / n
        return (variance_env - variance_within) / variance_total


def grouped_variance(Y, groups):
    """Sum over groups of the unbiased within-group sum of squares, per column of Y.

    Every group must have at least two members.
    """
    codes, n_codes = _codes(groups, Y.device)
    ones = torch.ones((Y.shape[0], 1), dtype=Y.dtype, device=Y.device)
    sizes = _segment_sum(ones, codes, n_codes)
    within = _segment_sum(Y ** 2, codes, n_codes) - _segment_sum(Y, codes, n_codes) ** 2 / sizes
    return torch.sum(within * (sizes / (sizes - 1)), axis=0)


def remove_environment(Y, environment):
    """Subtract the mean of each environmental group, one environmental variable at a time.

    Returns a new tensor; Y is not modified.
    """
    environment = _as_environment(environment, Y.shape[0])
    ones = torch.ones((Y.shape[0], 1), dtype=Y.dtype, device=Y.device)
    for a in range(environment.shape[1]):
        Y = _center_by(Y, ones, *_codes(environment[:, a], Y.device))
    return Y


def anova_heritability(Y, groups, environment=None, return_variance=False,
                       env_adjusted_total=False):
    """ANOVA heritability of each column of the (n, k) tensor or array Y.

    groups: length-n labels of genetically related groups (e.g. clones or families).
    environment: (n, g) categorical environmental variables, or None.

    The heritability is (V_env - V_within) / V_total, where V_env is the variance left after
    removing environmental group means and V_within the within-group variance. With clonal groups
    this is broad-sense heritability; for groups with genetic relatedness r, divide by r for
    narrow-sense. Groups with a single member are dropped. With env_adjusted_total, the denominator
    is V_env. With return_variance, returns (genetic variance, total variance) per individual
    instead. For repeated calls on the same individuals, use AnovaDesign.

    A tensor gives tensors (differentiable in Y); an array is computed in float64 and gives arrays.
    """
    if isinstance(Y, torch.Tensor):
        design = AnovaDesign(groups, environment, Y.device)
        return design.heritability(Y, return_variance, env_adjusted_total)
    Y = torch.tensor(np.asarray(Y, dtype=np.float64))
    result = AnovaDesign(groups, environment).heritability(Y, return_variance, env_adjusted_total)
    return tuple(r.numpy() for r in result) if return_variance else result.numpy()


def genetic_covariance(Y, N, groups, environment=None, correlation=False):
    """Genetic covariance between each column of Y and N (one trait, or one per column of Y).

    Both are standardized first. Estimated from the genetic variances of Y, N and Y + N.
    With correlation, negative genetic variances are set to zero and the covariance is
    divided by the genetic standard deviations (0 where either variance is 0).
    """
    if N.dim() == 1:
        N = N.reshape((N.shape[0], 1))[:, np.zeros(Y.shape[1], dtype=int)]

    N = N - torch.mean(N, axis=0).reshape((1, -1))
    N = N / (torch.mean(N ** 2, axis=0) ** 0.5).reshape((1, -1))
    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))
    Y = Y / (torch.mean(Y ** 2, axis=0) ** 0.5).reshape((1, -1))

    var_Y, _ = anova_heritability(Y, groups, environment, return_variance=True)
    var_N, _ = anova_heritability(N, groups, environment, return_variance=True)
    var_YN, _ = anova_heritability(Y + N, groups, environment, return_variance=True)

    if correlation:
        var_Y[var_Y < 0] = 0
        var_N[var_N < 0] = 0
        var_YN[var_YN < 0] = 0

    covariance = (var_YN - var_Y - var_N) / 2

    if correlation:
        covariance = covariance / ((var_Y * var_N) ** 0.5)
        covariance[var_Y == 0] = 0
        covariance[var_N == 0] = 0

    return covariance


def _dummies(labels, drop_first=False):
    _, inverse = np.unique(labels, return_inverse=True)
    D = np.zeros((len(labels), inverse.max() + 1))
    D[np.arange(len(labels)), inverse] = 1
    return D[:, 1:] if drop_first else D


def _projection(X):
    """Orthogonal projection onto the column space of X (rank-deficient X allowed)."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, s > s[0] * 1e-10]
    return U @ U.T


class Henderson3:
    """Heritability by Henderson's Method III.

    The model is y = environment (fixed) + group + subgroup-within-group + residual.

    groups: length-n labels of genetically related groups (e.g. families).
    environment: (n, g) categorical environmental variables (fixed effects), or None.
    subgroups: optional length-n labels of units nested in groups that share a non-genetic effect
    (e.g. plots of one family); labels only need to be unique within a group.

    With P(.) the projection onto the column space of the given design matrices,
        Q_group    = P(env, group) - P(env)
        Q_subgroup = P(env, group, subgroup) - P(env, group)
        Q_residual = I - P(env, group, subgroup)
    and E[y'Q y] is linear in the variance components, which gives unbiased estimates however
    environment and groups are confounded. The (n, n) forms are computed once per design, so this
    suits a fixed set of individuals; heritability(Y) is then differentiable in Y.
    """

    def __init__(self, groups, environment=None, subgroups=None, device='cpu'):
        groups = np.asarray(groups).astype(str)
        n = len(groups)
        environment = _as_environment(environment, n)
        env_dummies = [_dummies(environment[:, a], True) for a in range(environment.shape[1])]
        E = np.concatenate([np.ones((n, 1))] + env_dummies, 1)
        Z = [_dummies(groups)]
        if subgroups is not None:
            subgroup_labels = np.char.add(np.char.add(groups, '|'),
                                          np.asarray(subgroups).astype(str))
            Z.append(_dummies(subgroup_labels))

        projections = [_projection(E)]
        design = E
        for Zi in Z:
            design = np.concatenate([design, Zi], 1)
            projections.append(_projection(design))
        Q = [projections[i + 1] - projections[i] for i in range(len(Z))]
        Q.append(np.eye(n) - projections[-1])

        # E[y'Q_i y] = sum_j tr(Q_i Z_j Z_j') s2_j + tr(Q_i) s2_residual
        ZZ = [Zi @ Zi.T for Zi in Z]
        C = np.array([[np.einsum('ij,ji->', Qi, ZZj) for ZZj in ZZ] + [np.trace(Qi)] for Qi in Q])
        self.Cinv = torch.tensor(np.linalg.inv(C), device=device, dtype=torch.float64)
        self.Q = torch.tensor(np.stack(Q), device=device, dtype=torch.float64)

    def components(self, Y):
        """Variance components of each column of Y: rows are group, (subgroup,) residual."""
        Y = Y.to(self.Q.dtype)
        if Y.dim() == 1:
            Y = Y[:, None]
        quadratic = torch.stack([((Qi @ Y) * Y).sum(0) for Qi in self.Q])
        return self.Cinv @ quadratic

    def heritability(self, Y, denominator='total'):
        """Group variance over a denominator.

        The denominator is the total variance of Y ('total', as anova_heritability) or the sum of
        the variance components ('components', the mixed-model convention).
        """
        components = self.components(Y)
        if denominator == 'total':
            Y = Y.to(self.Q.dtype)
            if Y.dim() == 1:
                Y = Y[:, None]
            return components[0] / Y.var(0)
        return components[0] / components.sum(0)

    def forms(self):
        """(n, n) arrays A and B with y'A y / y'B y = heritability(y) (denominator 'total').

        A gives the group variance, a fixed combination of the y'Q y; B the variance of y.
        """
        Q = self.Q.cpu().numpy()
        A = np.tensordot(self.Cinv[0].cpu().numpy(), Q, axes=1)
        n = Q.shape[1]
        return A, (np.eye(n) - 1.0 / n) / (n - 1)


ESTIMATORS = ('anova', 'henderson3')


def heritability(Y, groups, environment=None, subgroups=None, estimator='anova'):
    """Heritability of each column of the (n, k) array Y by an estimator of ESTIMATORS.

    'anova' is anova_heritability (subgroups must be None); 'henderson3' is
    Henderson3(groups, environment, subgroups).heritability. Computed in float64; returns an array.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if estimator == 'anova':
        if subgroups is not None:
            raise ValueError("subgroups need estimator='henderson3'")
        return anova_heritability(Y, groups, environment)
    if estimator == 'henderson3':
        return Henderson3(groups, environment, subgroups).heritability(torch.tensor(Y)).numpy()
    raise ValueError(f'estimator must be one of {ESTIMATORS}, not {estimator!r}')
