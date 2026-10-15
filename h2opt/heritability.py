"""Heritability of traits (ANOVA, Henderson's Method III), differentiable in PyTorch."""

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

    def heritability(self, Y):
        """See anova_heritability."""
        mask = self.keep(Y.dtype)
        n = mask.sum(axis=0)
        Y2 = (Y - (Y * mask).sum(axis=0) / n) * mask
        Y2 = Y2 / ((torch.abs(Y2) * mask).sum(axis=0) / n)
        variance_total = torch.sum(Y2 ** 2, axis=0)

        if len(self.environment) == 0:
            variance_env = variance_total
        else:
            Y2 = self._remove_environment(Y2, mask, n)
            variance_env = torch.sum(Y2 ** 2, axis=0)

        sums = _segment_sum(Y2, self.groups, self.n_groups)
        sums_sq = _segment_sum(Y2 ** 2, self.groups, self.n_groups)
        sizes = _segment_sum(mask, self.groups, self.n_groups)
        within = (sums_sq - sums ** 2 / sizes.clamp(min=1)) * (sizes / (sizes - 1).clamp(min=1))
        variance_within = torch.sum(within, axis=0)

        return (variance_env - variance_within) / variance_total

    def _remove_environment(self, Y, mask, n):
        """Centered Y (0 outside mask) minus the means of each environmental variable in turn,
        centered again."""
        for codes, n_codes in self.environment:
            Y = _center_by(Y, mask, codes, n_codes)
        return (Y - Y.sum(axis=0) / n) * mask

    def forms(self):
        """(n, n) arrays A and B with y'A y / y'B y = heritability(y), for a design without rows.

        B gives the total variance of y, A the variance left after removing the environment minus
        the within-group variance (both times the number of individuals that count).
        """
        if self._keep.shape[1] != 1:
            raise ValueError('forms needs a design of all individuals (no rows)')
        mask = self.keep(torch.float64)
        n = mask.sum(axis=0)
        identity = torch.eye(mask.shape[0], dtype=torch.float64, device=mask.device)
        centered = (identity - mask.T / n) * mask
        adjusted = self._remove_environment(centered, mask, n)
        sizes = _segment_sum(mask, self.groups, self.n_groups)[self.groups]
        within = _center_by(adjusted, mask, self.groups, self.n_groups)
        within = within * (sizes / (sizes - 1).clamp(min=1)) ** 0.5
        A = adjusted.T @ adjusted - within.T @ within
        return A.cpu().numpy(), (centered.T @ centered).cpu().numpy()


def anova_heritability(Y, groups, environment=None):
    """ANOVA heritability of each column of the (n, k) tensor or array Y.

    groups: length-n labels of genetically related groups (e.g. clones or families).
    environment: (n, g) categorical environmental variables, or None.

    The heritability is (V_env - V_within) / V_total, where V_env is the variance left after
    removing environmental group means and V_within the within-group variance. With clonal groups
    this is broad-sense heritability; for groups with genetic relatedness r, divide by r for
    narrow-sense. Groups with a single member are dropped. For repeated calls on the same
    individuals, use AnovaDesign.

    A tensor gives tensors (differentiable in Y); an array is computed in float64 and gives arrays.
    """
    if isinstance(Y, torch.Tensor):
        design = AnovaDesign(groups, environment, Y.device)
        return design.heritability(Y)
    Y = torch.tensor(np.asarray(Y, dtype=np.float64))
    return AnovaDesign(groups, environment).heritability(Y).numpy()


def _dummies(labels, drop_first=False):
    _, inverse = np.unique(labels, return_inverse=True)
    D = np.zeros((len(labels), inverse.max() + 1))
    D[np.arange(len(labels)), inverse] = 1
    return D[:, 1:] if drop_first else D


def _range_basis(X):
    """Orthonormal basis of the column space of X (rank-deficient X allowed)."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > s[0] * 1e-10]


class Henderson3:
    """Heritability by Henderson's Method III.

    The model is y = environment (fixed) + group + subgroup-within-group + residual.

    groups: length-n labels of genetically related groups (e.g. families).
    environment: (n, g) categorical environmental variables (fixed effects), or None.
    subgroups: optional length-n labels of units nested in groups that share a non-genetic effect
    (e.g. plots of one family); labels only need to be unique within a group.
    rows: optional (n, k) boolean array; as in AnovaDesign, column j of Y is then a trait of the
    individuals where column j of rows is true (the other entries are ignored).

    With P(.) the projection onto the column space of the given design matrices,
        Q_group    = P(env, group) - P(env)
        Q_subgroup = P(env, group, subgroup) - P(env, group)
        Q_residual = I - P(env, group, subgroup)
    and E[y'Q y] is linear in the variance components, which gives unbiased estimates however
    environment and groups are confounded. Each projection is kept as an orthonormal basis U of
    its range (y'P y = |U'y|^2), computed once per set of rows; heritability(Y) is then
    differentiable in Y.
    """

    def __init__(self, groups, environment=None, subgroups=None, device='cpu', rows=None):
        groups = np.asarray(groups).astype(str)
        n = len(groups)
        environment = _as_environment(environment, n)
        if subgroups is not None:
            subgroups = np.char.add(np.char.add(groups, '|'), np.asarray(subgroups).astype(str))
        rows = np.ones((n, 1), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        # one decomposition per distinct set of rows
        row_sets, self._set_of_column = np.unique(rows.T, axis=0, return_inverse=True)
        self._set_of_column = self._set_of_column.reshape(-1)
        decompositions = [self._decompose(groups[r], environment[r],
                                          None if subgroups is None else subgroups[r])
                          for r in row_sets]
        self.n_components = decompositions[0][1].shape[0]
        self._sets = []
        for s, (r, (bases, Cinv)) in enumerate(zip(row_sets, decompositions, strict=True)):
            padded = []
            for U in bases:
                full = np.zeros((n, U.shape[1]))
                full[r] = U
                padded.append(torch.tensor(full, device=device, dtype=torch.float64))
            columns = torch.tensor(np.flatnonzero(self._set_of_column == s), device=device)
            self._sets.append((torch.tensor(r, device=device, dtype=torch.float64), padded,
                               torch.tensor(Cinv, device=device, dtype=torch.float64), columns))
        # with one column of rows (or none), it serves every column of Y; with k columns, Y has
        # k columns
        self._one_set = rows.shape[1] == 1
        self._mask = (torch.tensor(rows, device=device, dtype=torch.float64)
                      if rows.shape[1] > 1 or not rows.all() else None)

    @staticmethod
    def _decompose(groups, environment, subgroups):
        """Bases of P(env), P(env, group)(, P(env, group, subgroup)) of these individuals, and the
        inverse of the matrix C with E[y'Q_i y] = sum_j C_ij s2_j (s2: group, (subgroup,)
        residual)."""
        n = len(groups)
        E = np.concatenate([np.ones((n, 1))] + [_dummies(environment[:, a], True)
                                               for a in range(environment.shape[1])], 1)
        Z = [_dummies(groups)] + ([] if subgroups is None else [_dummies(subgroups)])
        bases = [_range_basis(E)]
        design = E
        for Zi in Z:
            design = np.concatenate([design, Zi], 1)
            bases.append(_range_basis(design))
        # tr(Q_i Z_j Z_j') and tr(Q_i) from the bases: tr(U U' M M') = |U'M|^2
        projected = [[np.sum((U.T @ Zj) ** 2) for Zj in Z] + [U.shape[1]] for U in bases]
        full = [np.sum(Zj ** 2) for Zj in Z] + [n]
        C = np.array([np.subtract(projected[i + 1], projected[i]) for i in range(len(Z))]
                     + [np.subtract(full, projected[-1])])
        return bases, np.linalg.inv(C)

    def components(self, Y):
        """Variance components of each column of Y: rows are group, (subgroup,) residual."""
        Y = Y.to(torch.float64)
        if Y.dim() == 1:
            Y = Y[:, None]
        if self._one_set:
            mask, bases, Cinv, _ = self._sets[0]
            return Cinv @ self._quadratic(Y * mask[:, None], bases)
        components = torch.zeros((self.n_components, Y.shape[1]), dtype=Y.dtype, device=Y.device)
        for mask, bases, Cinv, columns in self._sets:
            Ys = Y[:, columns] * mask[:, None]
            components = components.index_copy(1, columns, Cinv @ self._quadratic(Ys, bases))
        return components

    @staticmethod
    def _quadratic(Y, bases):
        """y'Q y of each column for Q_group, (Q_subgroup,) Q_residual: differences of y'P y over
        P(env), P(env, group), ..., and y'y."""
        norms = [((U.T @ Y) ** 2).sum(0) for U in bases] + [(Y ** 2).sum(0)]
        return torch.stack([norms[i + 1] - norms[i] for i in range(len(bases))])

    def heritability(self, Y):
        """Group variance over the total variance of Y (as anova_heritability)."""
        components = self.components(Y)
        Y = Y.to(torch.float64)
        if Y.dim() == 1:
            Y = Y[:, None]
        mask = self._sets[0][0][:, None] if self._mask is None else self._mask
        n = mask.sum(0)
        mean = (Y * mask).sum(0) / n
        variance = (((Y - mean) * mask) ** 2).sum(0) / (n - 1)
        return components[0] / variance

    def forms(self):
        """(n, n) arrays A and B with y'A y / y'B y = heritability(y), for a
        design without rows.

        A gives the group variance, a fixed combination of the y'Q y; B the variance of y.
        """
        if self._mask is not None:
            raise ValueError('forms needs a design of all individuals (no rows)')
        _, bases, Cinv, _ = self._sets[0]
        P = [(U @ U.T).cpu().numpy() for U in bases]
        n = P[0].shape[0]
        Q = [P[i + 1] - P[i] for i in range(len(P) - 1)] + [np.eye(n) - P[-1]]
        A = np.tensordot(Cinv[0].cpu().numpy(), np.stack(Q), axes=1)
        return A, (np.eye(n) - 1.0 / n) / (n - 1)


ESTIMATORS = ('anova', 'henderson3')


def heritability_design(estimator, groups, environment=None, subgroups=None, device='cpu',
                        rows=None):
    """The design of an estimator of ESTIMATORS: AnovaDesign for 'anova' (subgroups must be None)
    or Henderson3 for 'henderson3'. Raises ValueError for any other estimator."""
    if estimator == 'anova':
        if subgroups is not None:
            raise ValueError("subgroups need estimator='henderson3'")
        return AnovaDesign(groups, environment, device, rows=rows)
    if estimator == 'henderson3':
        return Henderson3(groups, environment, subgroups, device, rows=rows)
    raise ValueError(f'estimator must be one of {ESTIMATORS}, not {estimator!r}')


def heritability(Y, groups, environment=None, subgroups=None, estimator='anova'):
    """Heritability of each column of the (n, k) array Y by an estimator of ESTIMATORS
    (heritability_design). Computed in float64; returns an array."""
    Y = torch.tensor(np.asarray(Y, dtype=np.float64))
    return heritability_design(estimator, groups, environment, subgroups).heritability(Y).numpy()
