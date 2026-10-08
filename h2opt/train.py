"""Training synthetic traits to maximize heritability."""

import contextlib
import copy

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from .decorrelation import Decorrelation, orthonormal_basis
from .heritability import heritability, heritability_design
from .models import LinearConvModel, LinearModel, TraitModels


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


def _constant(learning_rate, n_iter):
    return np.full(n_iter, learning_rate)


def _adam(parameters, learning_rate, capturable):
    return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.99, 0.999),
                            capturable=capturable)


def _rmsprop(parameters, learning_rate, capturable):
    return torch.optim.RMSprop(parameters, lr=learning_rate, capturable=capturable)


# The optimizer of one trait and the learning rate of each of its steps, by name.
_OPTIMIZERS = {'adam': (_adam, _cosine), 'rmsprop': (_rmsprop, _constant)}

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


def train_batch(models, X, groups, environment, train_rows, noise_sd, earlier=None,
                n_iter=10000, optimizer='adam', learning_rate=1e-3, noise='normal', clip=None,
                device='cpu', subgroups=None, estimator='anova', max_copies=None):
    """Train one trait in each of several models at once; each copy is trained as by train.

    models: B modules of one class, each mapping (n, m) measurements to an (n, 1) trait; they are
    trained in place. train_rows: (B, n) boolean array, the training individuals of each copy.
    noise_sd: length B, the noise standard deviation of each copy. earlier: None, or B arrays
    (n, t_b) with the earlier traits of each copy for all individuals; in the loss, the trait of a
    copy is its residual after least-squares regression on its earlier traits on its training rows.
    All copies share each noise draw, scaled by their own noise_sd, so differences between
    their noise are not confounded with the draw. The loss is the sum of the negative
    heritabilities of the copies, and each copy's parameters get only its own gradient, so the
    result does not depend on which copies are trained together. With max_copies, at most that
    many copies are trained at once (e.g. to fit large models in GPU memory), each chunk with its
    own noise draws. A model can draw more noise in its forward pass in training mode (e.g.
    LinearConvModel); these draws are different for each copy. The models are trained in training
    mode and returned in evaluation mode. The other options are as in train. Returns models.
    """
    n_copies = len(models)
    if max_copies is not None and n_copies > max_copies:
        train_rows = np.asarray(train_rows, dtype=bool).reshape((n_copies, -1))
        noise_sd = np.broadcast_to(np.asarray(noise_sd, dtype=float), (n_copies,))
        for start in range(0, n_copies, max_copies):
            chunk = slice(start, start + max_copies)
            train_batch(models[chunk], X, groups, environment, train_rows[chunk],
                        noise_sd[chunk], None if earlier is None else earlier[chunk],
                        n_iter, optimizer, learning_rate, noise, clip, device, subgroups,
                        estimator)
        return models
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"optimizer must be 'adam' or 'rmsprop', not {optimizer!r}")
    if noise not in _NOISE:
        raise ValueError(f"noise must be 'uniform' or 'normal', not {noise!r}")
    train_rows = np.asarray(train_rows, dtype=bool).reshape((n_copies, -1))
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    if earlier is None:
        earlier = [np.zeros((len(groups), 0))] * n_copies
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    noise_sd = torch.as_tensor(np.asarray(noise_sd, dtype=np.float32), device=device)
    draw = _NOISE[noise]

    # only the individuals that some copy trains on take part
    used = np.any(train_rows, axis=0)
    train_rows = train_rows[:, used]
    X = X[torch.as_tensor(used, device=device)]
    design = heritability_design(estimator, groups[used],
                                 environment[used] if environment is not None else None,
                                 None if subgroups is None else np.asarray(subgroups)[used],
                                 device, train_rows.T)
    # residuals and their scale are over all training rows; the heritability drops singleton groups
    mask = torch.tensor(train_rows.T, dtype=torch.float32, device=device)
    n_train = mask.sum(axis=0)
    bases = _bases([np.asarray(E)[used] for E in earlier], train_rows, device)

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
            if noise == 'normal':
                # each copy's Z w is normal with standard deviation |w| per individual, so draw it
                # directly: the same loss distribution without a noisy (n, m) draw
                noise_w = draw((X.shape[0], 1), device=device) * torch.linalg.norm(W, axis=0)
            else:
                # (X + sd Z) w = X w + sd (Z w): no noisy copy of X per copy
                noise_w = draw(X.shape, device=device) @ W
            return X @ W + noise_w * noise_sd.reshape((1, -1)) + params['lin1.bias'].T
        Z = draw(X.shape, device=device)
        sd = noise_sd.reshape((-1,) + (1,) * X.dim())
        return vmap(forward, randomness='different')(params, buffers, X + Z * sd).T

    def step():
        Y = noisy_traits()
        Y = (Y - (Y * mask).sum(axis=0) / n_train) * mask
        Y = Y - torch.einsum('knt,kt->nk', bases, torch.einsum('knt,nk->kt', bases, Y))
        Y = Y / ((Y ** 2 * mask).sum(axis=0) / n_train) ** 0.5
        if clip is not None:
            Y = torch.clamp(Y, -clip, clip)
        loss = -torch.sum(design.heritability(Y))
        loss.backward()
        trait_optimizer.step()

    # On a GPU, one step is recorded as a CUDA graph after a few ordinary steps and then replayed:
    # a step launches many small kernels, and one replay costs less than launching them.
    cuda = torch.device(device).type == 'cuda'
    make_optimizer, schedule = _OPTIMIZERS[optimizer]
    rates = schedule(learning_rate, n_iter)
    rate = torch.tensor(learning_rate, device=device) if cuda else learning_rate
    trait_optimizer = make_optimizer(list(params.values()), rate, cuda)

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
          optimizer='adam', learning_rate=1e-3, noise_sd=0.1, noise='normal',
          clip=None, device='cpu', verbose=True, subgroups=None, estimator='anova',
          max_copies=None):
    """Train synthetic traits one at a time to maximize their heritability on training data.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Traits 0..n_traits-1 are trained in order. In the loss, trait t is the residual of its output
    after least-squares regression on the outputs of traits 0..t-1 on the training individuals, so
    the test individuals never shape the objective. The final traits are
    synthetic_traits(model, X, train_test == 0); with verbose, the train and test heritability of
    each trait is printed when it is trained.
    Each step adds noise to the training measurements and maximizes the heritability.
    noise_sd is one noise standard deviation for all traits or one per trait (length n_traits).
    optimizer is 'adam' (the default: Adam with momentum 0.99, and a learning rate that decays from
    learning_rate to 0 on a cosine schedule over the n_iter steps of each trait) or 'rmsprop'
    (RMSprop with a constant learning rate, the paper's optimizer). The heritability does not
    change with the scale of a trait, so RMSprop's steps shrink relative to the weights as the
    weights grow, and it converges slowly for correlated measurements; momentum and the decay fix
    this. noise is 'normal' (standard deviation noise_sd, the default) or 'uniform' (the paper's
    noise, in [0, noise_sd)). With clip, the standardized traits are clipped to [-clip, clip] in the
    loss; by default (None) the loss is the plain heritability.
    estimator is 'anova' (anova_heritability) or 'henderson3' (Henderson3 with subgroups, e.g.
    plots of one family; h2opt.heritability.ESTIMATORS); it is optimized and reported.
    If model_file is given, the whole model is saved after each trait. This is train_models with
    one model.
    """
    train_models([model], X, groups, environment, np.asarray(train_test)[None],
                 model_files=None if model_file is None else [model_file], n_traits=n_traits,
                 n_iter=n_iter, optimizer=optimizer,
                 learning_rate=learning_rate, noise_sd=noise_sd, noise=noise, clip=clip,
                 device=device, verbose=verbose, subgroups=subgroups, estimator=estimator,
                 max_copies=max_copies)
    return model


def train_models(models, X, groups, environment, train_test, model_files=None, n_traits=1,
                 n_iter=10000, optimizer='adam', learning_rate=1e-3,
                 noise_sd=0.1, noise='normal', clip=None, device='cpu', verbose=True,
                 subgroups=None, estimator='anova', max_copies=None):
    """Train several TraitModels at once, each as by train with its own data split and noise.

    models: B TraitModels of one kind. train_test: (B, n) array, the train_test of each model.
    noise_sd: one noise standard deviation for all, one per model (B, 1), or one per model and
    trait (B, n_traits). model_files: None, or one file per model. Trait t of all models is trained
    together by train_batch, each model on top of its own traits 0..t-1, so the models share each
    noise draw. With verbose, the train and test heritability of each trait of each model is
    printed. The other options are as in train. Returns models, in evaluation mode.
    """
    groups = np.asarray(groups)
    subgroups = None if subgroups is None else np.asarray(subgroups)
    train_test = np.asarray(train_test).reshape((len(models), -1))
    is_train, is_test = train_test == 0, train_test == 1
    sd = np.broadcast_to(np.asarray(noise_sd, dtype=float), (len(models), n_traits))
    env = np.asarray(environment) if environment is not None else None
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    for model in models:
        model.to(device)

    for trait in range(n_traits):
        earlier = []
        for model in models:
            with torch.no_grad():
                earlier.append(model(x, np.arange(trait)).double().cpu().numpy())
        train_batch([model.models[trait] for model in models], X, groups, environment, is_train,
                    sd[:, trait], earlier, n_iter=n_iter, optimizer=optimizer,
                    learning_rate=learning_rate, noise=noise, clip=clip, device=device,
                    subgroups=subgroups, estimator=estimator, max_copies=max_copies)
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
                             max_copies=None):
    """Train several TraitModels of LinearConvModel in two stages, each with its own data split.

    The first stage gives the start of the second stage:
    1. A TraitModels of LinearModel on the flattened X, trained by train_models for n_linear_iter
       steps with input noise of standard deviation linear_noise_sd.
    2. A TraitModels of LinearConvModel(make_conv(), w / s, b / s, ...) per model, with the
       weight w and bias b of each linear trait. s is the standard deviation of the output of
       that linear trait on the training individuals of the model, so each trait starts at the
       linear trait with unit standard deviation. It is trained by train_models for n_iter steps
       with no shared input noise: the linear branch draws noise of standard deviation
       linear_noise_sd (as input noise), conv draws input noise of standard deviation
       conv_noise_sd.

    make_conv: a function with no arguments that returns a new convolutional model, e.g.
    functools.partial(ImageConvModel, n_channels, image_size). X: the input of the convolutional
    model, with individuals on the first axis. train_test: (B, n) array, the train_test of each of
    the B models. linear_noise_sd, conv_noise_sd: as noise_sd in train_models. Both stages train
    trait t of all models together on top of traits 0..t-1 of the same stage. Both use the
    default optimizer of train (Adam). The other options are as in train_models. Returns the B
    trained models.
    """
    X = np.asarray(X)
    flat = X.reshape((len(X), -1))
    train_test = np.asarray(train_test).reshape((-1, len(X)))
    n_models = len(train_test)
    linear_sd = np.broadcast_to(np.asarray(linear_noise_sd, dtype=float), (n_models, n_traits))
    conv_sd = np.broadcast_to(np.asarray(conv_noise_sd, dtype=float), (n_models, n_traits))
    options = dict(n_traits=n_traits, device=device, verbose=verbose, subgroups=subgroups,
                   estimator=estimator, max_copies=max_copies)

    linear = [TraitModels(n_traits, LinearModel, flat.shape[1]) for _ in range(n_models)]
    train_models(linear, flat, groups, environment, train_test, n_iter=n_linear_iter,
                 noise_sd=linear_sd, **options)

    models = []
    for b in range(n_models):
        model = TraitModels(n_traits, make_conv)
        for t in range(n_traits):
            w = linear[b].models[t].lin1.weight.detach().cpu().numpy()
            bias = linear[b].models[t].lin1.bias.detach().cpu().numpy()
            s = (flat[train_test[b] == 0] @ w.T).std()
            if s == 0:
                raise ValueError(f'linear trait {t} of model {b} is constant on its training '
                                 'individuals')
            model.models[t] = LinearConvModel(model.models[t], w / s, bias / s, linear_sd[b, t],
                                              conv_sd[b, t])
        models.append(model)
    return train_models(models, X, groups, environment, train_test, n_iter=n_iter, noise_sd=0.0,
                        **options)
