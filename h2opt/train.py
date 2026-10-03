"""Training synthetic traits to maximize heritability."""

import numpy as np
import torch

from .decorrelation import Decorrelation, orthonormal_basis, residualize
from .heritability import AnovaDesign, anova_heritability


def synthetic_traits(model, X, is_train, traits=None):
    """Synthetic traits of all individuals: the outputs of the selected traits of model (default
    all), made uncorrelated on the training individuals (is_train, boolean) by Decorrelation.

    X: (n, m) array or tensor. Returns an (n, k) float64 array.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        Y = model(torch.as_tensor(X, dtype=torch.float32, device=device), traits).cpu().numpy()
    return Decorrelation().fit(Y[np.asarray(is_train, bool)]).transform(Y)


def _loss_traits(Y, basis, clip):
    """Residual of Y on the earlier traits (basis), standardized and clipped to [-clip, clip]."""
    Y = residualize(Y, basis)
    Y = Y / torch.mean(Y ** 2, axis=0) ** 0.5
    return Y if clip is None else torch.clamp(Y, -clip, clip)


def train(model, X, groups, environment, train_test, model_file=None, n_traits=1, first_trait=0,
          n_iter=10000, learning_rate=1e-4, noise_level=0.1, noise='uniform', clip=2.0,
          penalty=None, device='cpu', verbose=True, print_every=100, save_every=1000):
    """Train synthetic traits one at a time to maximize their ANOVA heritability on training data.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Traits first_trait..n_traits-1 are trained in order (traits before first_trait are taken as
    already trained). In the loss, trait t is the residual of its output after least-squares
    regression on the outputs of traits 0..t-1 on the training individuals, so the test
    individuals never shape the objective. The final traits are
    synthetic_traits(model, X, train_test == 0); with verbose, their train and test heritability
    is printed every print_every steps.
    Each step adds noise to the training measurements and maximizes the mean heritability with
    RMSprop. noise is 'uniform' (in [0, noise_level), the paper's sorghum setting) or 'normal'
    (standard deviation noise_level, the paper's simulation setting). The standardized traits are
    clipped to [-clip, clip] in the loss (clip=None: no clipping; the simulation used none).
    penalty(trait_model, Y) is an optional regularization term added to the loss, given the model
    of the current trait and its raw (n_train, 1) output on the noisy training data.
    If model_file is given, the whole model is saved every save_every steps and after each trait.
    """
    X = torch.tensor(X).float().to(device)
    model.to(device)
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    train_test = np.asarray(train_test)
    is_train, is_test = train_test == 0, train_test == 1

    def labels(rows):
        return groups[rows], environment[rows] if environment is not None else None

    if noise == 'uniform':
        def draw_noise(shape):
            return torch.rand(size=shape, device=device)
    elif noise == 'normal':
        def draw_noise(shape):
            return torch.randn(size=shape, device=device)
    else:
        raise ValueError(f"noise must be 'uniform' or 'normal', not {noise!r}")

    train_design = AnovaDesign(*labels(is_train), device)
    X_train_clean = X[is_train]

    for trait in range(first_trait, n_traits):
        with torch.no_grad():
            earlier = model(X_train_clean, np.arange(trait)).double()
        basis = orthonormal_basis(earlier).float()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        for step in range(n_iter):
            X_train = X_train_clean + draw_noise(X_train_clean.shape) * noise_level

            Y_raw = model(X_train, np.array([trait]))
            Y = _loss_traits(Y_raw, basis, clip)
            loss = -1 * torch.mean(train_design.heritability(Y))
            if penalty is not None:
                loss = loss + penalty(model.models[trait], Y_raw)

            if verbose and step % print_every == 0:
                Y = synthetic_traits(model, X, is_train, np.arange(trait + 1))[:, -1:]
                message = f'trait {trait} step {step}: train heritability '
                message += f'{anova_heritability(Y[is_train], *labels(is_train))}'
                if np.any(is_test):
                    message += ', test heritability '
                    message += f'{anova_heritability(Y[is_test], *labels(is_test))}'
                print(message)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_file is not None and step % save_every == 0:
                torch.save(model, model_file)

        if model_file is not None:
            torch.save(model, model_file)

    return model
