"""ANOVA heritability and genetic covariance of traits, differentiable in PyTorch."""

import numpy as np
import torch
import torch.nn.functional as F


def _group_index(labels):
    """Sort order of `labels` and the start/end positions (inclusive) of each group in that order."""
    order = np.argsort(labels)
    _, start, counts = np.unique(labels[order], return_index=True, return_counts=True)
    return order, start, start + counts - 1


def _as_environment(environment, n):
    """Environment as an (n, g) array of categorical labels; None means no environmental variables."""
    if environment is None:
        return np.zeros((n, 0))
    environment = np.asarray(environment)
    if environment.ndim == 1:
        environment = environment.reshape((-1, 1))
    if environment.shape[0] == 0:
        return np.zeros((n, 0))
    return environment


def grouped_variance(Y, groups):
    """Sum over groups of the unbiased within-group sum of squares, per column of Y.

    Every group must have at least two members.
    """
    order, start, end = _group_index(groups)

    Y_sorted = Y[order]
    if Y_sorted.dim() == 2:
        Y_sorted = F.pad(Y_sorted, (0, 0, 1, 0), "constant", 0)
    else:
        Y_sorted = F.pad(Y_sorted, (1, 0), "constant", 0)

    Y_cumsum = torch.cumsum(Y_sorted, dim=0)
    Y_sq_cumsum = torch.cumsum(Y_sorted ** 2, dim=0)

    sums = Y_cumsum[end + 1] - Y_cumsum[start]
    sums_sq = Y_sq_cumsum[end + 1] - Y_sq_cumsum[start]
    sizes = torch.tensor(end + 1 - start).float().to(Y.device)

    within = sums_sq - ((sums ** 2) / sizes.reshape((-1, 1)))
    within = within * (sizes / (sizes - 1)).reshape((-1, 1))
    return torch.sum(within, axis=0)


def remove_environment(Y, environment):
    """Subtract the mean of each environmental group, one environmental variable at a time.

    Returns a new tensor; Y is not modified.
    """
    environment = _as_environment(environment, Y.shape[0])
    Y = Y.clone()

    for a in range(environment.shape[1]):
        order, start, end = _group_index(environment[:, a])

        Y_sorted = torch.cat((torch.zeros((1, Y.shape[1])).to(Y.device), Y[order]))
        Y_cumsum = torch.cumsum(Y_sorted, dim=0)
        sums = Y_cumsum[end + 1] - Y_cumsum[start]

        sizes_np = end + 1 - start
        sizes = torch.tensor(sizes_np).float().to(Y.device)
        means = sums / sizes.reshape((-1, 1))

        # group index of each row in sorted order
        group_of_row = np.zeros(order.shape[0])
        group_of_row[np.cumsum(sizes_np)[:-1]] = 1
        group_of_row = np.cumsum(group_of_row)

        Y[order] = Y[order] - means[group_of_row]

    return Y


def anova_heritability(Y, groups, environment=None, return_variance=False, env_adjusted_total=False):
    """ANOVA heritability of each column of the (n, k) tensor Y.

    groups: length-n labels of genetically related groups (e.g. clones or families).
    environment: (n, g) categorical environmental variables, or None.

    The heritability is (V_env - V_within) / V_total, where V_env is the variance left after removing
    environmental group means and V_within the within-group variance. With clonal groups this is
    broad-sense heritability; for groups with genetic relatedness r, divide by r for narrow-sense.
    Groups with a single member are dropped. With env_adjusted_total, the denominator is V_env.
    With return_variance, returns (genetic variance, total variance) per individual instead.
    """
    groups = np.asarray(groups)
    environment = _as_environment(environment, len(groups))

    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if np.min(counts) == 1:
        keep = np.argwhere(counts[inverse] >= 2)[:, 0]
        return anova_heritability(Y[keep], groups[keep], environment[keep], return_variance=return_variance,
                                  env_adjusted_total=env_adjusted_total)

    if return_variance:
        Y2 = Y
    else:
        Y2 = Y - torch.mean(Y, axis=0).reshape((1, -1))
        Y2 = Y2 / (torch.mean(torch.abs(Y2), axis=0)).reshape((1, -1))

    variance_total = torch.sum((Y2 - torch.mean(Y2, axis=0)) ** 2, axis=0)

    if environment.shape[1] == 0:
        variance_env = variance_total
    else:
        Y2 = remove_environment(Y2, environment)
        variance_env = torch.sum((Y2 - torch.mean(Y2, axis=0).reshape((1, -1))) ** 2, axis=0)

    variance_within = grouped_variance(Y2, groups)

    if env_adjusted_total:
        variance_total = variance_env

    if return_variance:
        return (variance_env - variance_within) / Y.shape[0], variance_total / Y.shape[0]
    return (variance_env - variance_within) / variance_total


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
    """Heritability by Henderson's Method III for y = environment (fixed) + group + subgroup-within-group + residual.

    groups: length-n labels of genetically related groups (e.g. families).
    environment: (n, g) categorical environmental variables (fixed effects), or None.
    subgroups: optional length-n labels of units nested in groups that share a non-genetic effect
    (e.g. plots of one family); labels only need to be unique within a group.

    With P(.) the projection onto the column space of the given design matrices,
        Q_group    = P(env, group) - P(env)
        Q_subgroup = P(env, group, subgroup) - P(env, group)
        Q_residual = I - P(env, group, subgroup)
    and E[y'Q y] is linear in the variance components, which gives unbiased estimates however environment and
    groups are confounded. The (n, n) forms are computed once per design, so this suits a fixed set of
    individuals; heritability(Y) is then differentiable in Y.
    """

    def __init__(self, groups, environment=None, subgroups=None, device='cpu'):
        groups = np.asarray(groups).astype(str)
        n = len(groups)
        environment = _as_environment(environment, n)
        E = np.concatenate([np.ones((n, 1))] + [_dummies(environment[:, a], True) for a in range(environment.shape[1])], 1)
        Z = [_dummies(groups)]
        if subgroups is not None:
            Z.append(_dummies(np.char.add(np.char.add(groups, '|'), np.asarray(subgroups).astype(str))))

        projections = [_projection(E)]
        design = E
        for Zi in Z:
            design = np.concatenate([design, Zi], 1)
            projections.append(_projection(design))
        Q = [projections[i + 1] - projections[i] for i in range(len(Z))] + [np.eye(n) - projections[-1]]

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
        """Group variance over the total variance of Y ('total', as anova_heritability) or over the sum of the
        variance components ('components', the mixed-model convention)."""
        components = self.components(Y)
        if denominator == 'total':
            Y = Y.to(self.Q.dtype)
            if Y.dim() == 1:
                Y = Y[:, None]
            return components[0] / Y.var(0)
        return components[0] / components.sum(0)
