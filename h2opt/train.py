"""Training synthetic traits to maximize heritability."""

import numpy as np
import torch

from .heritability import anova_heritability


def decorrelate(Y, clip=None):
    """Standardize the columns of Y and make each orthogonal to the columns before it (Gram-Schmidt).

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
    """Remove from each column of Y its projection on the (standardized) columns of background, then standardize.

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


def _synthetic_traits(model, X, trait, background):
    Y = model(X, np.array([trait]))
    Y = project_out(Y, background)
    return decorrelate(Y, clip=2)


def train(model, X, groups, environment, train_test, model_file=None, n_traits=1, first_trait=0, n_iter=10000,
          learning_rate=1e-4, noise_level=0.1, verbose=True, print_every=100, save_every=10):
    """Train synthetic traits one at a time to maximize their ANOVA heritability on the training set.

    model: a TraitModels with at least n_traits traits; X: (n, m) measurements.
    groups, environment: as in anova_heritability. train_test: length-n array, 0 = train, 1 = test.
    Trait t is trained after traits first_trait..t-1 and is made orthogonal to all earlier traits.
    Each step adds uniform noise in [0, noise_level) to the training measurements and maximizes the
    mean heritability with RMSprop. If model_file is given, the whole model is saved every save_every steps.
    Returns the model.
    """
    X = torch.tensor(X).float()
    groups = np.asarray(groups)
    environment = np.asarray(environment) if environment is not None else None
    train_test = np.asarray(train_test)
    is_train = train_test == 0
    has_test = np.any(train_test == 1)

    def subset(rows):
        return groups[rows], (environment[rows] if environment is not None else None)

    for trait in range(first_trait, n_traits):
        if trait > 0:
            background = decorrelate(model(X, np.arange(trait)).detach())
        else:
            background = torch.zeros((X.shape[0], 0))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        for step in range(n_iter):
            X_train = X[is_train]
            X_train = X_train + torch.rand(size=X_train.shape) * noise_level

            Y = _synthetic_traits(model, X_train, trait, background[is_train])
            loss = -1 * torch.mean(anova_heritability(Y, *subset(is_train)))

            if verbose and step % print_every == 0:
                with torch.no_grad():
                    Y = _synthetic_traits(model, X, trait, background)
                    message = f'trait {trait} step {step}: train heritability {anova_heritability(Y[is_train], *subset(is_train)).numpy()}'
                    if has_test:
                        is_test = train_test == 1
                        message += f', test heritability {anova_heritability(Y[is_test], *subset(is_test)).numpy()}'
                print(message)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_file is not None and step % save_every == 0:
                torch.save(model, model_file)

    return model
