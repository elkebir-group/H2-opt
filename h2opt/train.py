"""Training synthetic traits to maximize heritability."""

import copy

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from .decorrelation import Decorrelation, orthonormal_basis
from .heritability import AnovaDesign, anova_heritability
from .models import LinearModel


def synthetic_traits(model, X, is_train, traits=None):
    """Synthetic traits of all individuals: the outputs of the selected traits of model (default
    all), made uncorrelated on the training individuals (is_train, boolean) by Decorrelation.

    X: (n, m) array or tensor. Returns an (n, k) float64 array.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        Y = model(torch.as_tensor(X, dtype=torch.float32, device=device), traits).cpu().numpy()
    return Decorrelation().fit(Y[np.asarray(is_train, bool)]).transform(Y)


def _adam(parameters, learning_rate, n_iter):
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=(0.99, 0.999))
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)


def _rmsprop(parameters, learning_rate, n_iter):
    optimizer = torch.optim.RMSprop(parameters, lr=learning_rate)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


# The optimizer of one trait and its learning-rate schedule, by name.
_OPTIMIZERS = {'adam': _adam, 'rmsprop': _rmsprop}

# Noise draws of standard scale, by name: uniform in [0, 1) or standard normal.
_NOISE = {'uniform': torch.rand, 'normal': torch.randn}


def _bases(earlier, rows, device):
    """(k, n, t) orthonormal bases of the earlier traits of k copies, each on its own training rows
    ((k, n) boolean) and zero on the other rows, padded with zero columns to the largest number t
    of earlier traits."""
    t = max(np.asarray(E).shape[1] for E in earlier)
    bases = torch.zeros((len(earlier), rows.shape[1], t), device=device)
    for c, E in enumerate(earlier):
        basis = orthonormal_basis(torch.as_tensor(np.asarray(E, dtype=np.float64)[rows[c]],
                                                  device=device)).float()
        bases[c, torch.as_tensor(rows[c], device=device), :basis.shape[1]] = basis
    return bases


def train_batch(models, X, groups, environment, train_rows, noise_levels, earlier=None,
                n_iter=10000, optimizer='adam', learning_rate=1e-3, noise='normal', clip=None,
                device='cpu'):
    """Train one trait in each of several models at once; each copy is trained as by train.

    models: B modules of one class, each mapping (n, m) measurements to an (n, 1) trait; they are
    trained in place. train_rows: (B, n) boolean array, the training individuals of each copy.
    noise_levels: length B, the noise level of each copy. earlier: None, or B arrays (n, t_b) with
    the earlier traits of each copy for all individuals; in the loss, the trait of a copy is its
    residual after least-squares regression on its earlier traits on its training rows.
    All copies share each noise draw, scaled by their own noise level, so differences between
    their noise levels are not confounded with the draw. The loss is the sum of the negative
    heritabilities of the copies, and each copy's parameters get only its own gradient, so the
    result does not depend on which copies are trained together. The other options are as in
    train. Returns models.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be 'adam' or 'rmsprop', not {optimizer!r}")
    if noise not in _NOISE:
        raise ValueError(f"noise must be 'uniform' or 'normal', not {noise!r}")
    n_copies = len(models)
    train_rows = np.asarray(train_rows, dtype=bool).reshape((n_copies, -1))
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    if earlier is None:
        earlier = [np.zeros((len(groups), 0))] * n_copies
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    noise_levels = torch.as_tensor(np.asarray(noise_levels, dtype=np.float32), device=device)
    draw = _NOISE[noise]

    # only the individuals that some copy trains on take part
    used = np.any(train_rows, axis=0)
    train_rows = train_rows[:, used]
    X = X[torch.as_tensor(used, device=device)]
    design = AnovaDesign(groups[used], environment[used] if environment is not None else None,
                         device, rows=train_rows.T)
    # residuals and their scale are over all training rows; the heritability drops singleton groups
    mask = torch.tensor(train_rows.T, dtype=torch.float32, device=device)
    n_train = mask.sum(axis=0)
    bases = _bases([np.asarray(E)[used] for E in earlier], train_rows, device)

    for model in models:
        model.to(device)
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to('meta')

    def forward(p, b, x):
        return functional_call(base, (p, b), (x,))[:, 0]

    def noisy_traits(Z):
        """(n, B) outputs of the copies on X + level * Z."""
        if isinstance(base, LinearModel):
            # (X + level Z) w = X w + level (Z w): no noisy copy of X per copy
            W = params['lin1.weight'][:, 0, :].T
            return X @ W + (Z @ W) * noise_levels.reshape((1, -1)) + params['lin1.bias'].T
        return vmap(forward)(params, buffers, X + Z * noise_levels.reshape((-1, 1, 1))).T

    trait_optimizer, schedule = _OPTIMIZERS[optimizer](list(params.values()), learning_rate,
                                                       n_iter)
    for _ in range(n_iter):
        Y = noisy_traits(draw(X.shape, device=device))
        Y = (Y - (Y * mask).sum(axis=0) / n_train) * mask
        Y = Y - torch.einsum('knt,kt->nk', bases, torch.einsum('knt,nk->kt', bases, Y))
        Y = Y / ((Y ** 2 * mask).sum(axis=0) / n_train) ** 0.5
        if clip is not None:
            Y = torch.clamp(Y, -clip, clip)
        loss = -torch.sum(design.heritability(Y))
        trait_optimizer.zero_grad()
        loss.backward()
        trait_optimizer.step()
        schedule.step()

    with torch.no_grad():
        for c, model in enumerate(models):
            for name, parameter in model.named_parameters():
                parameter.copy_(params[name][c])
    return models


def train(model, X, groups, environment, train_test, model_file=None, n_traits=1, first_trait=0,
          n_iter=10000, optimizer='adam', learning_rate=1e-3, noise_level=0.1, noise='uniform',
          clip=None, device='cpu', verbose=True):
    """Train synthetic traits one at a time to maximize their ANOVA heritability on training data.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Traits first_trait..n_traits-1 are trained in order (traits before first_trait are taken as
    already trained). In the loss, trait t is the residual of its output after least-squares
    regression on the outputs of traits 0..t-1 on the training individuals, so the test
    individuals never shape the objective. The final traits are
    synthetic_traits(model, X, train_test == 0); with verbose, the train and test heritability of
    each trait is printed when it is trained.
    Each step adds noise to the training measurements and maximizes the heritability.
    noise_level is one level for all traits or one per trait (length n_traits).
    optimizer is 'adam' (the default: Adam with momentum 0.99, and a learning rate that decays from
    learning_rate to 0 on a cosine schedule over the n_iter steps of each trait) or 'rmsprop'
    (RMSprop with a constant learning rate, the paper's optimizer). The heritability does not
    change with the scale of a trait, so RMSprop's steps shrink relative to the weights as the
    weights grow, and it converges slowly for correlated measurements; momentum and the decay fix
    this. noise is 'uniform' (in [0, noise_level), the paper's sorghum setting) or 'normal'
    (standard deviation noise_level, the paper's simulation setting). With clip, the standardized
    traits are clipped to [-clip, clip] in the loss (the paper's sorghum setting was 2); by
    default (None) the loss is the plain heritability, as in the paper's simulation.
    If model_file is given, the whole model is saved after each trait. This is train_models with
    one model.
    """
    train_models([model], X, groups, environment, np.asarray(train_test)[None],
                 model_files=None if model_file is None else [model_file], n_traits=n_traits,
                 first_trait=first_trait, n_iter=n_iter, optimizer=optimizer,
                 learning_rate=learning_rate, noise_levels=noise_level, noise=noise, clip=clip,
                 device=device, verbose=verbose)
    return model


def train_models(models, X, groups, environment, train_test, model_files=None, n_traits=1,
                 first_trait=0, n_iter=10000, optimizer='adam', learning_rate=1e-3,
                 noise_levels=0.1, noise='uniform', clip=None, device='cpu', verbose=True):
    """Train several TraitModels at once, each as by train with its own data split and noise.

    models: B TraitModels of one kind. train_test: (B, n) array, the train_test of each model.
    noise_levels: one level for all, one per model (B, 1), or one per model and trait
    (B, n_traits). model_files: None, or one file per model. Trait t of all models is trained
    together by train_batch, each model on top of its own traits 0..t-1, so the models share each
    noise draw. With verbose, the train and test heritability of each trait of each model is
    printed. The other options are as in train. Returns models.
    """
    groups = np.asarray(groups)
    train_test = np.asarray(train_test).reshape((len(models), -1))
    is_train, is_test = train_test == 0, train_test == 1
    levels = np.broadcast_to(np.asarray(noise_levels, dtype=float), (len(models), n_traits))
    env = np.asarray(environment) if environment is not None else None
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    for model in models:
        model.to(device)

    for trait in range(first_trait, n_traits):
        earlier = []
        for model in models:
            with torch.no_grad():
                earlier.append(model(x, np.arange(trait)).double().cpu().numpy())
        train_batch([model.models[trait] for model in models], X, groups, environment, is_train,
                    levels[:, trait], earlier, n_iter=n_iter, optimizer=optimizer,
                    learning_rate=learning_rate, noise=noise, clip=clip, device=device)
        for b, model in enumerate(models):
            if verbose:
                Y = synthetic_traits(model, X, is_train[b], np.arange(trait + 1))[:, -1:]
                message = f'model {b} ' if len(models) > 1 else ''
                message += f'trait {trait}: train heritability {_h2(Y, groups, env, is_train[b])}'
                if np.any(is_test[b]):
                    message += f', test heritability {_h2(Y, groups, env, is_test[b])}'
                print(message, flush=True)
            if model_files is not None:
                torch.save(model, model_files[b])

    return models


def _h2(Y, groups, environment, rows):
    """ANOVA heritability of the traits Y on the selected rows."""
    return anova_heritability(Y[rows], groups[rows],
                              environment[rows] if environment is not None else None)
