"""Training synthetic traits to maximize heritability."""

import numpy as np
import torch

from .heritability import AnovaDesign


def decorrelate(Y, clip=None):
    """Standardize the columns of Y; make each orthogonal to the columns before it (Gram-Schmidt).

    The projection coefficients are computed without gradient, so each column only receives gradient
    through itself. If clip is given, values are clipped to [-clip, clip] after standardizing.
    """
    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))
    Y = Y / (torch.mean(Y ** 2, axis=0) ** 0.5).reshape((1, -1))

    if Y.shape[1] > 1:
        Y_basis = Y.detach().clone()
        for a in range(1, Y_basis.shape[1]):
            for b in range(a):
                cor = torch.mean(Y_basis[:, a] * Y_basis[:, b])
                Y_basis[:, a] = Y_basis[:, a] - (cor * Y_basis[:, b])
                Y_basis[:, a] = Y_basis[:, a] / (torch.mean(Y_basis[:, a] ** 2) ** 0.5)

        # project each column on the orthonormal basis of the columns before it
        cor_all = torch.matmul(Y_basis.T, Y) / Y.shape[0]
        upper = torch.triu(torch.ones((Y.shape[1], Y.shape[1])), diagonal=1).to(Y.device)
        Y = Y - torch.matmul(Y_basis, cor_all * upper)

    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))
    Y = Y / (torch.mean(Y ** 2, axis=0) ** 0.5).reshape((1, -1))

    if clip is not None:
        Y = torch.clamp(Y, -clip, clip)
    return Y


def project_out(Y, background):
    """Project the (standardized) columns of background out of each column of Y, then standardize.

    background is assumed to have orthogonal columns (e.g. the output of decorrelate).
    """
    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))

    if background.shape[1] > 0:
        background = background - torch.mean(background, axis=0).reshape((1, -1))
        background = background / (torch.mean(background ** 2, axis=0).reshape((1, -1)) ** 0.5)
        coef = torch.mean(Y * background, axis=0)
        Y = Y - torch.sum(coef.reshape((1, -1)) * background, axis=1).reshape((-1, 1))

    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))
    return Y / (torch.mean(Y ** 2, axis=0).reshape((1, -1)) ** 0.5)


def _synthetic_traits(Y, background, clip):
    return decorrelate(project_out(Y, background), clip=clip)


def train(model, X, groups, environment, train_test, model_file=None, n_traits=1, first_trait=0,
          n_iter=10000, learning_rate=1e-4, noise_level=0.1, noise='uniform', clip=2.0,
          penalty=None, device='cpu', verbose=True, print_every=100, save_every=1000):
    """Train synthetic traits one at a time to maximize their ANOVA heritability on training data.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Trait t is trained after traits first_trait..t-1 and is made orthogonal to all earlier traits.
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

    def design(rows):
        env = environment[rows] if environment is not None else None
        return AnovaDesign(groups[rows], env, device)

    if noise == 'uniform':
        def draw_noise(shape):
            return torch.rand(size=shape, device=device)
    elif noise == 'normal':
        def draw_noise(shape):
            return torch.randn(size=shape, device=device)
    else:
        raise ValueError(f"noise must be 'uniform' or 'normal', not {noise!r}")

    train_design = design(is_train)
    test_design = design(is_test) if np.any(is_test) else None
    X_train_clean = X[is_train]

    for trait in range(first_trait, n_traits):
        if trait > 0:
            background = decorrelate(model(X, np.arange(trait)).detach())
        else:
            background = torch.zeros((X.shape[0], 0), device=device)

        background_train = background[is_train]
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        for step in range(n_iter):
            X_train = X_train_clean + draw_noise(X_train_clean.shape) * noise_level

            Y_raw = model(X_train, np.array([trait]))
            Y = _synthetic_traits(Y_raw, background_train, clip)
            loss = -1 * torch.mean(train_design.heritability(Y))
            if penalty is not None:
                loss = loss + penalty(model.models[trait], Y_raw)

            if verbose and step % print_every == 0:
                with torch.no_grad():
                    Y = _synthetic_traits(model(X, np.array([trait])), background, clip)
                    train_h = train_design.heritability(Y[is_train]).cpu().numpy()
                    message = f'trait {trait} step {step}: train heritability {train_h}'
                    if test_design is not None:
                        test_h = test_design.heritability(Y[is_test]).cpu().numpy()
                        message += f', test heritability {test_h}'
                print(message)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_file is not None and step % save_every == 0:
                torch.save(model, model_file)

        if model_file is not None:
            torch.save(model, model_file)

    return model
