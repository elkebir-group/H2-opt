"""Training synthetic traits to maximize heritability."""

import contextlib
import copy

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from .decorrelation import Decorrelation, orthonormal_basis
from .heritability import heritability, heritability_design
from .models import LinearConvModel, LinearModel, TraitModels
from .spatial import papadakis_torch


def synthetic_traits(model, X, is_train, traits=None):
    """Synthetic traits of all individuals: the outputs of the selected traits of model (default
    all), made uncorrelated on the training individuals (is_train, boolean) by Decorrelation.

    X: (n, m) array or tensor. The model is put in evaluation mode, so it draws no noise.
    Returns an (n, k) float64 array.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        Y = model(torch.as_tensor(X, dtype=torch.float32, device=device), traits).cpu().numpy()
    return Decorrelation().fit(Y[np.asarray(is_train, bool)]).transform(Y)


def _cosine(learning_rate, n_iter):
    """Learning rate of each step, from learning_rate to 0 on a cosine schedule."""
    return learning_rate * (1 + np.cos(np.pi * np.arange(n_iter) / n_iter)) / 2




def _l1(params):
    return sum(p.abs().flatten(1).sum(axis=1) for p in params.values())


def _l2(params):
    return sum((p ** 2).flatten(1).sum(axis=1) for p in params.values())


def _tv(params):
    # the weight matrices only: differences between neighboring features (the last axis)
    return sum((p[..., 1:] - p[..., :-1]).abs().flatten(1).sum(axis=1)
               for p in params.values() if p.dim() > 2)


# Penalties on the stacked parameters of B copies, by name: (B,) values and the power of the
# trait's spread that each is divided by, so that the penalty does not change with the scale of
# the trait. l1: sum of absolute values; l2: sum of squares; tv: total variation, the sum of the
# absolute differences between the weights of neighboring features.
_PENALTIES = {'l1': (_l1, 1), 'l2': (_l2, 2), 'tv': (_tv, 1)}


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


def train_batch(models, X, groups, environment, train_rows, noise_sd, earlier=None,
                n_iter=10000, learning_rate=1e-3, penalty=None, device='cpu', subgroups=None,
                estimator='anova', max_copies=None, neighbors=None):
    """Train one trait in each of several models at once; each copy is trained as by train.

    models: B modules of one class, each mapping (n, m) measurements to an (n, 1) trait; they are
    trained in place. train_rows: (B, n) boolean array, the training individuals of each copy.
    noise_sd: length B, the standard deviation of the normal input noise of each copy. earlier:
    None, or B arrays (n, t_b) with the earlier traits of each copy for all individuals; in the
    loss, the trait of a copy is its residual after least-squares regression on its earlier traits
    on its training rows. All copies share each noise draw, scaled by their own noise_sd, so
    differences between their noise are not confounded with the draw. The loss is the sum of the
    negative heritabilities of the copies, and each copy's parameters get only its own gradient.
    With max_copies, at most that many copies are trained at once (e.g. to fit large models in GPU
    memory); every chunk starts from the same random state and draws the noise of all n individuals,
    so every copy gets the same input noise whatever max_copies is. Without noise (all noise_sd 0)
    there is no draw. A model can draw more noise in its forward pass in training mode (e.g.
    LinearConvModel); all copies of a chunk share these draws, but they depend on the individuals of
    the chunk (the union of their training rows), so copies with different training rows that must
    share them belong in one chunk. The models are trained in training mode and returned in
    evaluation mode. The other options are as in train. Returns models.
    """
    n_copies = len(models)
    train_rows = np.asarray(train_rows, dtype=bool).reshape((n_copies, -1))
    noise_sd = np.broadcast_to(np.asarray(noise_sd, dtype=float), (n_copies,))
    size = n_copies if max_copies is None else max_copies
    seed = int(torch.randint(2**62, (1,)))
    cuda = torch.device(device).type == 'cuda'
    devices = [torch.device(device).index or 0] if cuda else []
    for start in range(0, n_copies, size):
        chunk = slice(start, start + size)
        with torch.random.fork_rng(devices):
            torch.manual_seed(seed)
            _train_batch(models[chunk], X, groups, environment, train_rows[chunk],
                         noise_sd[chunk], None if earlier is None else earlier[chunk], n_iter,
                         learning_rate, penalty, device, subgroups, estimator, neighbors)
    return models


def _train_batch(models, X, groups, environment, train_rows, noise_sd, earlier, n_iter,
                 learning_rate, penalty, device, subgroups, estimator, neighbors):
    """train_batch on one chunk of copies, all trained at once."""
    n_copies = len(models)
    if penalty is not None and penalty[0] not in _PENALTIES:
        raise ValueError(f"penalty must be 'l1', 'l2' or 'tv', not {penalty[0]!r}")
    train_rows = np.asarray(train_rows, dtype=bool).reshape((n_copies, -1))
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    if earlier is None:
        earlier = [np.zeros((len(groups), 0))] * n_copies
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    has_noise = bool(np.any(np.asarray(noise_sd)))
    noise_sd = torch.as_tensor(np.asarray(noise_sd, dtype=np.float32), device=device)

    # only the individuals that some copy trains on take part
    used = np.any(train_rows, axis=0)
    n_all = len(used)
    train_rows = train_rows[:, used]
    used_rows = torch.as_tensor(np.flatnonzero(used), device=device)
    X = X[used_rows]

    def draw(shape):
        """Standard normal noise of the used individuals, (n_used,) + shape. The draw is over all
        individuals, so each individual gets the same noise whatever the other copies of the
        chunk train on."""
        return torch.randn((n_all,) + shape, device=device)[used_rows]
    design = heritability_design(estimator, groups[used],
                                 environment[used] if environment is not None else None,
                                 None if subgroups is None else np.asarray(subgroups)[used],
                                 device, train_rows.T)
    # residuals and their scale are over all training rows; the heritability drops singleton groups
    mask = torch.tensor(train_rows.T, dtype=torch.float32, device=device)
    n_train = mask.sum(axis=0)
    bases = _bases([np.asarray(E)[used] for E in earlier], train_rows, device)
    if neighbors is not None:
        neighbors = torch.as_tensor(np.asarray(neighbors, dtype=np.float32)[np.ix_(used, used)],
                                    device=device)

    for model in models:
        model.to(device)
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to('meta').train()

    def forward(p, b, x):
        return functional_call(base, (p, b), (x,))[:, 0]

    def noisy_traits():
        """(n, B) outputs of the copies on X + noise_sd * Z, Z a fresh noise draw."""
        if isinstance(base, LinearModel):
            W = params['lin1.weight'][:, 0, :].T
            # (X + sd Z) w = X w + sd (Z w), and each copy's Z w is normal with standard deviation
            # |w| per individual, so draw it directly: no noisy (n, m) copy of X per copy
            noise_w = draw((1,)) * torch.linalg.norm(W, axis=0)
            return X @ W + noise_w * noise_sd.reshape((1, -1)) + params['lin1.bias'].T
        if not has_noise:
            # all copies see the same X: no draw, and no copy of X per copy
            return vmap(forward, in_dims=(0, 0, None), randomness='same')(params, buffers, X).T
        Z = draw(tuple(X.shape[1:]))
        sd = noise_sd.reshape((-1,) + (1,) * X.dim())
        return vmap(forward, randomness='same')(params, buffers, X + Z * sd).T

    def step():
        Y = noisy_traits()
        Y = (Y - (Y * mask).sum(axis=0) / n_train) * mask
        spread = Y.abs().sum(axis=0) / n_train
        Y = Y - torch.einsum('knt,kt->nk', bases, torch.einsum('knt,nk->kt', bases, Y))
        Y = Y / ((Y ** 2 * mask).sum(axis=0) / n_train) ** 0.5
        if neighbors is not None:
            Y = papadakis_torch(Y, neighbors, mask)
        loss = -torch.sum(design.heritability(Y))
        if penalty is not None:
            kind, weight = penalty
            values, power = _PENALTIES[kind]
            loss = loss + weight * torch.sum(values(params) / spread ** power)
        loss.backward()
        trait_optimizer.step()

    # On a GPU, one step is recorded as a CUDA graph after a few ordinary steps and then replayed:
    # a step launches many small kernels, and one replay costs less than launching them.
    cuda = torch.device(device).type == 'cuda'
    rates = _cosine(learning_rate, n_iter)
    rate = torch.tensor(learning_rate, device=device) if cuda else learning_rate
    trait_optimizer = torch.optim.Adam(list(params.values()), lr=rate, betas=(0.99, 0.999),
                                       capturable=cuda)

    def set_rate(i):
        if cuda:
            rate.fill_(rates[i])
        else:
            for group in trait_optimizer.param_groups:
                group['lr'] = rates[i]

    n_ordinary = min(n_iter, 3) if cuda else n_iter
    stream = torch.cuda.Stream(device) if cuda else None
    if cuda:
        stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream) if cuda else contextlib.nullcontext():
        for i in range(n_ordinary):
            set_rate(i)
            trait_optimizer.zero_grad(set_to_none=True)
            step()
    if cuda:
        torch.cuda.current_stream(device).wait_stream(stream)
    if n_ordinary < n_iter:
        graph = torch.cuda.CUDAGraph()
        trait_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            step()
        for i in range(n_ordinary, n_iter):
            set_rate(i)
            graph.replay()

    with torch.no_grad():
        for c, model in enumerate(models):
            for name, parameter in model.named_parameters():
                parameter.copy_(params[name][c])
            model.eval()
    return models


def train(model, X, groups, environment, train_test, model_file=None, n_traits=1, n_iter=10000,
          learning_rate=1e-3, noise_sd=0.1, penalty=None, device='cpu', verbose=True,
          subgroups=None, estimator='anova', max_copies=None, neighbors=None):
    """Train synthetic traits one at a time to maximize their heritability on training data.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Traits 0..n_traits-1 are trained in order. In the loss, trait t is the residual of its output
    after least-squares regression on the outputs of traits 0..t-1 on the training individuals, so
    the test individuals never shape the objective. The final traits are
    synthetic_traits(model, X, train_test == 0); with verbose, the train and test heritability of
    each trait is printed when it is trained.
    Each step adds normal noise to the training measurements and maximizes the heritability.
    noise_sd is one noise standard deviation for all traits or one per trait (length n_traits).
    The optimizer is Adam with momentum 0.99, and a learning rate that decays from learning_rate
    to 0 on a cosine schedule over the n_iter steps of each trait. The heritability does not
    change with the scale of a trait, so the steps of an optimizer without momentum (e.g. the
    paper's RMSprop) shrink relative to the weights as the weights grow, and it converges slowly
    for correlated measurements; momentum and the decay fix this.
    penalty: None (the default), or (kind, weight): weight times a penalty on the trainable
    parameters is added to the loss of each trait. kind is 'l1' (the sum of their absolute
    values), 'l2' (the sum of their squares) or 'tv' (total variation: the sum of the absolute
    differences between the weights of neighboring features, for the weight matrices only). The
    heritability does not change with the scale of a trait, so the l1 and tv penalties are divided
    by the spread of the trait (the mean absolute deviation of its noisy output on the training
    individuals, before the regression on the earlier traits), and l2 by its square.
    estimator is 'anova' (anova_heritability) or 'henderson3' (Henderson3 with subgroups, e.g.
    plots of one family; h2opt.heritability.ESTIMATORS); it is optimized and reported.
    neighbors: None (the default), or (n, n) nonnegative weights of the field neighbors of each
    individual (h2opt.spatial.papadakis); the loss is then the heritability of the trait after
    the nearest-neighbor adjustment, with neighbor means and slope over the training individuals
    only. The printed heritability is that of the unadjusted trait.
    If model_file is given, the whole model is saved after each trait. This is train_models with
    one model.
    """
    noise_sd = np.asarray(noise_sd, dtype=float)
    train_models([model], X, groups, environment, np.asarray(train_test)[None],
                 model_files=None if model_file is None else [model_file], n_traits=n_traits,
                 n_iter=n_iter, learning_rate=learning_rate,
                 noise_sd=noise_sd.reshape((1, -1)) if noise_sd.ndim else noise_sd,
                 penalty=penalty, device=device, verbose=verbose, subgroups=subgroups,
                 estimator=estimator, max_copies=max_copies, neighbors=neighbors)
    return model


def _per_model_and_trait(value, n_models, n_traits, name):
    """value (one for all, (n_models, 1) or (n_models, n_traits)) as an (n_models, n_traits)
    array. A 1-D value is refused: one value per model and one per trait look the same when
    n_models equals n_traits."""
    value = np.asarray(value, dtype=float)
    if value.ndim == 1 and value.size > 1:
        raise ValueError(f'{name} must be one value, ({n_models}, 1) or ({n_models}, {n_traits}), '
                         f'not a 1-D array of {value.size}')
    try:
        return np.broadcast_to(value.reshape(()) if value.ndim == 1 else value,
                               (n_models, n_traits))
    except ValueError:
        raise ValueError(f'{name} of shape {value.shape} does not fit {n_models} models and '
                         f'{n_traits} traits') from None


def train_models(models, X, groups, environment, train_test, model_files=None, n_traits=1,
                 n_iter=10000, learning_rate=1e-3, noise_sd=0.1, penalty=None, device='cpu',
                 verbose=True, subgroups=None, estimator='anova', max_copies=None,
                 neighbors=None, earlier=None):
    """Train several TraitModels at once, each as by train with its own data split and noise.

    models: B TraitModels of one kind. train_test: (B, n) array, the train_test of each model.
    noise_sd: one noise standard deviation for all, one per model (B, 1), or one per model and trait
    (B, n_traits); a 1-D array is refused, since one value per model and one per trait look the same
    when B equals n_traits. model_files: None, or one file per model. Trait t of all models is
    trained together by train_batch, each model on top of its own traits 0..t-1, so the models share
    each noise draw. earlier: None, or B arrays (n, t) of fixed traits of each model for all
    individuals (e.g. the traits of an earlier run); every trait is then trained on top of these as
    well as the model's own traits 0..t-1. With verbose, the train and test heritability of each
    trait of each model is printed (before the regression on earlier). The other options are as in
    train. Returns models, in evaluation mode.
    """
    groups = np.asarray(groups)
    subgroups = None if subgroups is None else np.asarray(subgroups)
    train_test = np.asarray(train_test).reshape((len(models), -1))
    is_train, is_test = train_test == 0, train_test == 1
    sd = _per_model_and_trait(noise_sd, len(models), n_traits, 'noise_sd')
    env = np.asarray(environment) if environment is not None else None
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    for model in models:
        model.to(device)

    for trait in range(n_traits):
        before = []
        for b, model in enumerate(models):
            with torch.no_grad():
                own = model(x, np.arange(trait)).double().cpu().numpy()
            before.append(own if earlier is None else np.hstack([np.asarray(earlier[b]), own]))
        train_batch([model.models[trait] for model in models], X, groups, environment, is_train,
                    sd[:, trait], before, n_iter=n_iter, learning_rate=learning_rate,
                    penalty=penalty,
                    device=device, subgroups=subgroups, estimator=estimator, max_copies=max_copies,
                    neighbors=neighbors)
        for b, model in enumerate(models):
            if verbose:
                Y = synthetic_traits(model, X, is_train[b], np.arange(trait + 1))[:, -1:]

                def h2(rows, Y=Y):
                    return heritability(Y[rows], groups[rows], None if env is None else env[rows],
                                        None if subgroups is None else subgroups[rows], estimator)

                message = f'model {b} ' if len(models) > 1 else ''
                message += f'trait {trait}: train heritability {h2(is_train[b])}'
                if np.any(is_test[b]):
                    message += f', test heritability {h2(is_test[b])}'
                print(message, flush=True)
            if model_files is not None:
                torch.save(model, model_files[b])

    for model in models:
        model.eval()
    return models


def train_linear_conv_models(make_conv, X, groups, environment, train_test, linear_noise_sd,
                             conv_noise_sd, n_traits=1, n_linear_iter=10000, n_iter=1000,
                             device='cpu', verbose=True, subgroups=None, estimator='anova',
                             max_copies=None, neighbors=None, earlier=None):
    """Train several TraitModels of LinearConvModel in two stages, each with its own data split.

    The first stage gives the start of the second stage:
    1. A TraitModels of LinearModel on the flattened X, trained by train_models for n_linear_iter
       steps with input noise of standard deviation linear_noise_sd. This stage does not depend on
       conv_noise_sd, so the models with the same train_test and linear_noise_sd share one linear
       TraitModels (e.g. the copies of one validation split at several levels of conv_noise_sd).
    2. A TraitModels of LinearConvModel(make_conv(), w / s, b / s, ...) per model, with the
       weight w and bias b of each linear trait. s is the standard deviation of the output of
       that linear trait on the training individuals of the model, so each trait starts at the
       linear trait with unit standard deviation, plus the output of a new convolutional model.
       The models with the same train_test start from the same make_conv() models (copies), so
       with the shared noise draws of train_batch they differ only by conv_noise_sd. The linear
       map and the convolutional model train together, by train_models for n_iter steps with no
       shared input noise: the linear branch draws noise of standard deviation linear_noise_sd
       (as input noise), conv draws input noise of standard deviation conv_noise_sd.

    make_conv: a function with no arguments that returns a new convolutional model, e.g.
    functools.partial(ImageConvModel, n_channels). X: the input of the convolutional
    model, with individuals on the first axis. train_test: (B, n) array, the train_test of each of
    the B models. linear_noise_sd, conv_noise_sd: as noise_sd in train_models. Both stages train
    trait t on top of traits 0..t-1 of the same stage. The first stage trains trait t of all
    linear models together. The second stage trains trait t of the models with the same
    train_test together, so a step computes only the training individuals of those models. Both
    use the optimizer of train. earlier: as in train_models, both stages train on
    top of these fixed traits. The other options are as in train_models. Returns the B trained
    models.
    """
    X = np.asarray(X)
    flat = X.reshape((len(X), -1))
    train_test = np.asarray(train_test).reshape((-1, len(X)))
    n_models = len(train_test)
    linear_sd = _per_model_and_trait(linear_noise_sd, n_models, n_traits, 'linear_noise_sd')
    conv_sd = _per_model_and_trait(conv_noise_sd, n_models, n_traits, 'conv_noise_sd')
    options = dict(n_traits=n_traits, device=device, verbose=verbose, subgroups=subgroups,
                   estimator=estimator, max_copies=max_copies, neighbors=neighbors)

    # one linear TraitModels per distinct (train_test, linear_sd), in order of first use
    shared = {}
    stage = [shared.setdefault((train_test[b].tobytes(), linear_sd[b].tobytes()), len(shared))
             for b in range(n_models)]
    first = [stage.index(i) for i in range(len(shared))]
    linear = [TraitModels(n_traits, LinearModel, flat.shape[1]) for _ in first]
    train_models(linear, flat, groups, environment, train_test[first], n_iter=n_linear_iter,
                 noise_sd=linear_sd[first],
                 earlier=None if earlier is None else [earlier[b] for b in first], **options)

    # the models with the same train_test start from the same convolutional models
    starts, models = {}, []
    for b in range(n_models):
        key = train_test[b].tobytes()
        if key not in starts:
            starts[key] = TraitModels(n_traits, make_conv)
        model = copy.deepcopy(starts[key])
        for t in range(n_traits):
            w = linear[stage[b]].models[t].lin1.weight.detach().cpu().numpy()
            bias = linear[stage[b]].models[t].lin1.bias.detach().cpu().numpy()
            s = (flat[train_test[b] == 0] @ w.T).std()
            if s == 0:
                raise ValueError(f'linear trait {t} of model {b} is constant on its training '
                                 'individuals')
            model.models[t] = LinearConvModel(model.models[t], w / s, bias / s, linear_sd[b, t],
                                              conv_sd[b, t])
        models.append(model)
    # the models with the same train_test train together, on their training individuals only
    splits = {}
    for b in range(n_models):
        splits.setdefault(train_test[b].tobytes(), []).append(b)
    for split in splits.values():
        train_models([models[b] for b in split], X, groups, environment, train_test[split],
                     n_iter=n_iter, noise_sd=0.0,
                     earlier=None if earlier is None else [earlier[b] for b in split], **options)
    return models
