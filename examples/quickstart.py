"""The README examples on the sorghum example data: heritability, training synthetic traits,
choosing the noise level and simulated spectra.

    uv run python examples/quickstart.py                 # as in the README (10,000 steps per trait)
    uv run python examples/quickstart.py --device cuda   # the same on a GPU
    uv run python examples/quickstart.py --steps 20      # a quick check
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from h2opt import (
    AutoEncoder,
    LinearModel,
    TraitModels,
    anova_heritability,
    encode_latent,
    latent_scale,
    load_npz,
    reconstruction_residuals,
    synthetic_traits,
    train,
)
from h2opt.folds import group_folds
from h2opt.selection import select_noise_level

EXAMPLES = Path(__file__).resolve().parent.parent / 'data' / 'examples'


def main(n_iter, device):
    # Data: sorghum hyperspectral measurements, clone labels and field sets.
    X = np.concatenate((load_npz(EXAMPLES / 'X_file1.npz'), load_npz(EXAMPLES / 'X_file2.npz')))
    groups = load_npz(EXAMPLES / 'genotypes.npz')
    environment = load_npz(EXAMPLES / 'environment.npz')
    # Hold out a third of the clones as the test set (0: training, 1: test individual).
    train_test = (group_folds(groups, n_folds=3) == 0).astype(int)
    is_train = train_test == 0

    # Heritability of the first 10 wavelengths.
    H = anova_heritability(torch.tensor(X[:, :10]).float(), groups, environment)
    print('heritability of wavelengths 1-10:', H.numpy().round(4))

    # Choose the noise level, then train 10 linear synthetic traits.
    noise_sd, _, _ = select_noise_level(lambda: LinearModel(X.shape[1]), X, groups,
                                             environment, subsets=[is_train],
                                             n_iter=n_iter, device=device)
    print('chosen noise standard deviation:', noise_sd[0].round(4))
    n_traits = 10
    model = TraitModels(n_traits, LinearModel, X.shape[1])
    train(model, X, groups, environment, train_test, n_traits=n_traits, n_iter=n_iter,
          noise_sd=noise_sd[0], device=device)
    traits = torch.tensor(synthetic_traits(model, X, is_train))  # n by n_traits
    for name, rows in (('training', is_train), ('test', ~is_train)):
        H = anova_heritability(traits[rows], groups[rows], environment[rows])
        print(f'{name} set heritability of the synthetic traits:', H.numpy().round(2))

    # Simulated spectra from latent traits.
    autoencoder = AutoEncoder.load(EXAMPLES / 'autoencoder.pt')
    latent = load_npz(EXAMPLES / 'simulatedLatentTraits.npz')
    latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)
    simulated = encode_latent(latent, autoencoder, latent_scale(autoencoder, X),
                                reconstruction_residuals(autoencoder, X),
                                np.random.RandomState(0))
    print('simulated spectra:', simulated.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000, help='training steps per trait')
    parser.add_argument('--device', default='cpu', help='torch device, e.g. cuda for a GPU')
    args = parser.parse_args()
    main(args.steps, args.device)
