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

## Calculating ANOVA heritability

Let n be the number of individuals.
`traits` is an n by k PyTorch tensor of phenotypes, `groups` a length-n array of labels of genetically related groups such as clones, and `environment` an n by g array of categorical environmental variables (or `None`).

```python
from h2opt import anova_heritability
H = anova_heritability(traits, groups, environment)
```

If groups are clones, this is the broad-sense heritability. If the individuals in a group have genetic relatedness r, the narrow-sense heritability is H / r.
Groups with a single individual are ignored.

`Henderson3(groups, environment, subgroups=None)` estimates heritability by Henderson's Method III, treating environment as fixed and groups (and optional subgroups nested in groups, e.g. plots of one family) as random effects.
Unlike `anova_heritability`, it stays unbiased when environment and groups are confounded, and it separates shared subgroup effects from genetic variance.
It precomputes n by n matrices for a fixed set of individuals, then `Henderson3(...).heritability(traits)` is differentiable.
`heritability(traits, groups, environment, subgroups, estimator)` computes either estimator (`ESTIMATORS`: `'anova'`, `'henderson3'`) on arrays.

Example: the heritability of the first 10 wavelengths of the sorghum hyperspectral data in `data/examples` (the measurements are split over two files because of GitHub file size limits).

```python
import numpy as np
import torch
from h2opt import load_npz, anova_heritability

X = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
groups = load_npz('data/examples/genotypes.npz')
environment = load_npz('data/examples/environment.npz')

H = anova_heritability(torch.tensor(X[:, :10]).float(), groups, environment)
# tensor([0.0530, 0.0322, 0.0545, 0.0715, 0.0774, 0.0755, 0.0564, 0.0422, 0.0479, 0.0503])
```

## Optimizing heritability

`train` learns synthetic traits one at a time. In the loss, each trait is its residual after least-squares regression on the earlier traits, on the training individuals.
`model` holds one model per trait (`TraitModels`), `X` is the n by m array of measurements, and `train_test` a length-n array with 0 for training and 1 for test individuals.
The heritability is only optimized on the training individuals.

Example: train a linear model extracting 10 synthetic traits from the sorghum data.

```python
import numpy as np
import torch
from h2opt import load_npz, synthetic_traits, train, TraitModels, LinearModel

X = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
groups = load_npz('data/examples/genotypes.npz')
environment = load_npz('data/examples/environment.npz')
train_test = np.zeros(X.shape[0], dtype=int)

n_traits = 10
model = TraitModels(n_traits, LinearModel, X.shape[1])
train(model, X, groups, environment, train_test, model_file='model.pt', n_traits=n_traits,
      n_iter=10000, noise_sd=0.005)

# n by n_traits synthetic traits, made uncorrelated on the training individuals
traits = synthetic_traits(model, X, train_test == 0)
```

Each step adds random noise to the training measurements (data augmentation), which regularizes the traits.

Options: `n_iter` steps per trait (default 10000); `optimizer`, `'adam'` (default) or `'rmsprop'`; `learning_rate` (default 1e-3); `noise_sd` for data augmentation (default 0.1), one for all traits or one per trait: the standard deviation of the normal noise added to the measurements at each step, or with `noise='uniform'` the maximum of the paper's uniform noise in [0, `noise_sd`); `clip` to clip the standardized traits in the loss to [-clip, clip] (default `None`: no clipping, so the loss is the plain heritability; the paper clipped sorghum at 2); `model_file` to save the model after each trait; and `device` (e.g. `"cuda"`) to train on a GPU.
`estimator` chooses the heritability that is optimized and reported: `'anova'` (default, `anova_heritability`) or `'henderson3'` (`Henderson3`, with `subgroups`, e.g. plots of one family).
`'adam'` uses momentum 0.99 and decays the learning rate to 0 on a cosine schedule over the `n_iter` steps of each trait.
`'rmsprop'` is the optimizer of the paper (constant learning rate). It converges slowly when the measurements are strongly correlated: the heritability does not change with the scale of a trait, so its steps shrink relative to the weights as the weights grow.
`ConvModel` is a convolutional alternative to `LinearModel` for spectra, and `ImageConvModel(n_channels, image_size)` one for multispectral images (input n x channels x height x width; the paper's Miscanthus model).

`LinearConvModel(conv, weight, bias, linear_noise_sd, conv_noise_sd)` is a fixed linear map of the flattened input (`weight`, `bias`) plus a trainable convolutional model `conv` whose output layer starts at zero; so the model starts at exactly the linear map, and only `conv` trains.
In training mode, each branch draws its own noise: the linear branch `linear_noise_sd` |w| per individual on its output (the same as input noise of that SD for a linear map), the convolutional model input noise of SD `conv_noise_sd`; so either branch can have more noise.
`train_linear_conv_models(make_conv, X, groups, environment, train_test, linear_noise_sd, conv_noise_sd)` trains it in two stages for several data splits: linear H2-opt with noise `linear_noise_sd` (`n_linear_iter` steps, default 10000), then the sum, started at each linear trait with unit standard deviation on the training individuals (`n_iter` steps, default 1000). In the second stage the linear branch keeps noise `linear_noise_sd` and the convolutional model gets input noise `conv_noise_sd`, each drawn by the model itself.

`train_batch` trains one trait in each of many models at once, each with its own training individuals, noise level and earlier traits.
`train_models` trains several `TraitModels` at once on it, each with its own `train_test` and noise levels (e.g. the outer folds of a dataset, or one model per noise level); `train` is `train_models` with one model.
The copies share each noise draw, scaled by their own level, and each copy gets only its own gradient. `max_copies` trains at most that many copies at once, for models too large to train all copies together on a GPU.
For `LinearModel` it is fast, because (X + noise) w = X w + noise w needs no noisy copy of X per model, and with normal noise each model's noise w is drawn directly (normal with standard deviation |w| per individual).
On a GPU, one step is recorded as a CUDA graph and replayed, which removes the cost of launching its many small kernels.
On an RTX 3080, one sorghum-sized linear model trains at about 2,500 steps per second, and 400 at about 210,000 model-steps per second.
Other models are trained with `torch.func.vmap`, which gives no speedup for `ConvModel`.

### Choosing the noise level

Every method chooses its regularization level by one procedure, `cross_validate_levels`: the groups are split into validation folds, the method fits its first trait at each level on the other folds, and the score is the heritability of that trait on the held-out fold. A method only supplies a function that fits all (subset, split, level) copies at once.
`select_noise_level` uses it to choose one noise level of H2-opt for all traits, and `PCH.tune` to choose the ridge of PCH, both by `best_level` (the best mean score).
The first trait sets the level: a mean over all traits rewards high noise, which spreads the heritable signal over more traits (the first traits lose heritability and the later traits gain it). The first trait also needs no earlier traits, so selection trains only it.
Each level is the standard deviation of normal noise as a fraction of `noise_scale(X)`, the root mean variance of the measurements, so the same levels apply to data on any scale.

```python
from h2opt.selection import select_noise_level

is_train = train_test == 0
noise_sd, scores, traits = select_noise_level(lambda: LinearModel(X.shape[1]), X, groups,
                                              environment, subsets=[is_train], n_iter=10000)
train(model, X, groups, environment, train_test, n_traits=n_traits, noise_sd=noise_sd[0])
```

Within each subset of the individuals (`subsets`, e.g. the training individuals of several outer folds, all chosen in one batch), the groups are split (`h2opt.folds.group_folds`) into 5 folds, and each fold is the held-out groups of one validation split (`n_folds`; `n_splits` uses only the first folds).
The levels are `NOISE_LEVELS` (0.001 to 100, quarter-decade steps), the one grid of every method. The chosen level has the best held-out heritability averaged over the splits.
All subsets, splits and levels are trained together by `train_batch`.
It returns the noise standard deviation of each subset for `train`: the chosen level times `noise_scale` of the subset's measurements.
Given a list of measurement sets of the same individuals (e.g. one per date), it trains and scores each set and chooses one level for all of them by their mean score; each set gets its own standard deviation, the level times its own `noise_scale`.
With `split_units` (e.g. the individuals themselves), the validation folds split those units instead of the groups.
`select_noise_level` also returns the scores (the chosen level is `best_level(levels, scores[i])`) and the trained validation traits for each subset, split and level. The traits of a level do not depend on the other levels, so another rule over the levels can be examined without training again.
Levels can also be scored in separate runs (e.g. to extend the grid later): `best_level(levels, scores)` applies the same rule to their scores stacked on the last axis.

For `LinearConvModel`, the linear branch keeps the level of linear H2-opt, and `score_linear_conv_levels` scores the first trait at each noise level of the convolutional branch on the same validation splits.
The chosen level is the highest level whose score is within one standard error of the best (`highest_level_within_one_se`). A high level makes the convolutional branch add almost nothing, so the model stays at the linear trait unless the convolutional branch clearly raises the held-out heritability.

## Baselines

`h2opt.baselines` has linear baselines with a common interface: `fit(X, groups, environment)` on the training individuals (`PCH.fit` also takes `subgroups` for `'henderson3'`), then `transform(X)`.
Their traits are centered and made uncorrelated on the training individuals (`Decorrelation`), like H2-opt's.

- `PCA(n_traits)`: principal components of the measurements.
- `GeneticPCA(n_traits)`: principal components of the ANOVA estimate of the genetic covariance.
- `PCH(n_traits, ridge, estimator='anova')`: principal components of heritability, maximizing the ANOVA heritability above (or, with `estimator='henderson3'` and `fit(X, groups, environment, subgroups)`, Henderson's Method III) with ridge regularization.
  Both estimators are ratios of quadratic forms in the trait, so the solution is exact. PCH works in the span of the training data, so the measurements may far outnumber the individuals (e.g. image pixels).
  `PCH.tune` chooses the ridge from `RIDGES` (1e-6 to 1e4), the squares of H2-opt's `NOISE_LEVELS`: for a linear trait, ridge s^2 is the penalty of noise level s. It uses the procedure and the rule of `select_noise_level` (`cross_validate_levels`, `best_level`: the best mean held-out heritability of the first trait, with the same estimator; `split_units` as there); given a list of measurement sets of the same individuals (e.g. one per date), it chooses one ridge for all of them by their mean score.
- `MaxHeritabilityFeatures(n_traits)`: the most heritable individual features, selected greedily.

```python
from h2opt.baselines import PCH

is_train = train_test == 0
pch = PCH(n_traits=5, ridge=1e-4).fit(X[is_train], groups[is_train], environment[is_train])
traits = pch.transform(X)
```

## Generating simulated hyperspectral measurements

`encode_latent` embeds latent traits into simulated spectra with a pretrained autoencoder.
Each latent trait, in standard units, is matched to the mean and spread of one latent dimension over the encodings of a reference set of real measurements (`latent_scale`).
The decoder is linearized at the latent means, so the decoded spectra are an exact linear function of the latent traits: the full tanh decoder adds products and squares of the latent traits, which are heritable extra traits.
Each simulated individual also gets the residual of a random real measurement (`reconstruction_residuals`: what the autoencoder does not reconstruct), so the spectra have the size and correlation of the real variation that the latent dimensions miss, and it is not heritable.
The pretrained autoencoder in `data/examples/autoencoder.pt` has 5 latent dimensions.

Example with the latent traits simulated with simplePHENOTYPES in the paper:

```python
import numpy as np
from h2opt import AutoEncoder, encode_latent, latent_scale, load_npz, reconstruction_residuals

autoencoder = AutoEncoder.load('data/examples/autoencoder.pt')
reference = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
latent = load_npz('data/examples/simulatedLatentTraits.npz')
latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)

X_simulated = encode_latent(latent, autoencoder, latent_scale(autoencoder, reference),
                           reconstruction_residuals(autoencoder, reference), np.random.RandomState(0))
```

## Tests

```bash
uv run --all-extras pytest                      # the suite
uv run --all-extras pytest tests/test_train.py  # narrow by path or -k while iterating
```

The CUDA tests are skipped when no GPU is present.

## Related repositories

- [`elkebir-group/H2-opt-analysis`](https://github.com/elkebir-group/H2-opt-analysis)
  (private): the analyses and figures of the paper (sorghum, Miscanthus, metabolomics,
  simulations). It installs this package from a sibling checkout.
