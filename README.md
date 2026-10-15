# H2-opt
H2-opt: A novel self-supervised algorithm to mine high-throughput phenotyping data for genetically-driven traits

<p align="center">
  <img width="400" height="220" src="./overview.png">
</p>

H2-opt learns synthetic traits from high-throughput phenotyping (HTP) measurements that maximize heritability.
It needs only the measurements and labels of genetically related groups (e.g. clones or families); no genotype data is used in training.

The code that reproduces the analyses and figures of the paper is in
[H2-opt-analysis](https://github.com/elkebir-group/H2-opt-analysis).

## Install

H2-opt needs Python 3.10+, PyTorch and numpy. The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras        # creates .venv with the package and the dev tools
uv run pre-commit install   # one-time, after cloning: runs ruff and codespell on commit
```

Prefix the commands below with `uv run`, or activate `.venv` first. Plain `pip install .` also
works if you only need the package.

## Examples

[`examples/quickstart.py`](examples/quickstart.py) runs all examples below on the sorghum example data
(`uv run python examples/quickstart.py`; add `--device cuda` to train on a GPU, or `--steps 20`
for a quick check).
The output of `uv run python examples/quickstart.py --device cuda` (about 1 minute on a GPU) is in
[`examples/quickstart_output.txt`](examples/quickstart_output.txt).

The examples below use the sorghum hyperspectral data in `data/examples` (the measurements are split over two files because of GitHub file size limits):

```python
import numpy as np
import torch
from h2opt import load_npz
from h2opt.folds import group_folds

X = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
groups = load_npz('data/examples/genotypes.npz')
environment = load_npz('data/examples/environment.npz')
train_test = (group_folds(groups, n_folds=3) == 0).astype(int)  # a third of the clones held out
is_train = train_test == 0
```

## Heritability

`anova_heritability(traits, groups, environment)` takes an n by k tensor of traits, a length-n array of group labels (e.g. clones) and an n by g array of categorical environmental variables (or `None`).
If groups are clones, this is the broad-sense heritability; if the individuals of a group have genetic relatedness r, the narrow-sense heritability is H / r.

```python
from h2opt import anova_heritability
H = anova_heritability(torch.tensor(X[:, :10]).float(), groups, environment)
# tensor([0.0530, 0.0322, 0.0545, 0.0715, 0.0774, 0.0755, 0.0564, 0.0422, 0.0479, 0.0503])
```

`Henderson3(groups, environment, subgroups)` estimates heritability by Henderson's Method III, with groups and optional nested subgroups (e.g. plots of one family) as random effects.
It stays unbiased when environment and groups are confounded. Both estimators are differentiable in the traits.

## Training synthetic traits

`train` learns synthetic traits one at a time, each maximizing heritability on the training individuals after regression on the earlier traits.
`synthetic_traits` returns them uncorrelated on the training individuals; their heritability on the held-out clones measures how well they generalize.

```python
from h2opt import synthetic_traits, train, TraitModels, LinearModel

n_traits = 10
model = TraitModels(n_traits, LinearModel, X.shape[1])
train(model, X, groups, environment, train_test, model_file='model.pt', n_traits=n_traits,
      n_iter=10000, noise_sd=0.005)
traits = torch.tensor(synthetic_traits(model, X, is_train))  # n by n_traits
H_test = anova_heritability(traits[~is_train], groups[~is_train], environment[~is_train])
```

Each step adds normal noise of standard deviation `noise_sd` to the measurements, which regularizes the traits.
Other options: `n_iter` steps per trait, `model_file` to save the model after each trait, `learning_rate`, `estimator` (`'anova'` or `'henderson3'`), `penalty` (L1, L2 or total variation on the parameters) and `device` (e.g. `"cuda"`).
Training uses Adam with a cosine learning-rate decay.

Models: `LinearModel`; `ConvModel` for spectra; `ImageConvModel` and `PositionImageConvModel` for multispectral images; and `LinearConvModel`, a linear map plus a CNN trained together, which `train_linear_conv_models` starts from linear H2-opt.
`train_models` trains several models at once (e.g. one per cross-validation fold).

## Choosing the noise level

`select_noise_level` chooses the noise level by cross-validation on the training individuals: it splits the groups into validation folds (4 partitions into 5 folds), scores the held-out heritability of the first trait at each level of `NOISE_LEVELS`, and takes the highest level within one standard error of the best (`highest_level_within_one_se`).
Levels are relative to `noise_scale(X)`, so the same grid applies to data on any scale.

```python
from h2opt.selection import select_noise_level

noise_sd, scores, _ = select_noise_level(lambda: LinearModel(X.shape[1]), X, groups, environment,
                                         subsets=[is_train], n_iter=10000)
train(model, X, groups, environment, train_test, n_traits=n_traits, noise_sd=noise_sd[0])
```

The same procedure chooses the ridge of PCH (`PCH.tune`) and the CNN level of `LinearConvModel` (`score_linear_conv_levels`).

## Baselines

`h2opt.baselines` has linear baselines with the interface `fit(X, groups, environment)` on the training individuals, then `transform(X)`; their traits are uncorrelated on the training individuals, as H2-opt's.

- `PCA(n_traits)`: principal components of the measurements.
- `GeneticPCA(n_traits)`: principal components of the genetic covariance.
- `PCH(n_traits, ridge)`: principal components of heritability, the exact linear maximum of heritability with a ridge.
- `MaxHeritabilityFeatures(n_traits)`: the most heritable individual measurements, selected greedily.

## Simulated hyperspectral measurements

`encode_latent` embeds latent traits into simulated spectra with a pretrained autoencoder (`data/examples/autoencoder.pt`, 5 latent dimensions).
The decoder is linearized at the latent means, and each simulated individual gets the non-heritable residual of a random real measurement.

```python
from h2opt import AutoEncoder, encode_latent, latent_scale, reconstruction_residuals

autoencoder = AutoEncoder.load('data/examples/autoencoder.pt')
latent = load_npz('data/examples/simulatedLatentTraits.npz')
latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)
simulated = encode_latent(latent, autoencoder, latent_scale(autoencoder, X),
                            reconstruction_residuals(autoencoder, X), np.random.RandomState(0))
```

## Tests

```bash
uv run --all-extras pytest
```

The CUDA tests are skipped when no GPU is present.
