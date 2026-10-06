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
      n_iter=10000, noise_level=0.005)

# n by n_traits synthetic traits, made uncorrelated on the training individuals
traits = synthetic_traits(model, X, train_test == 0)
```

Each step adds random noise to the training measurements (data augmentation), which regularizes the traits.
For a linear trait, noise with standard deviation s has the same effect as a ridge penalty s^2 |w|^2 in the denominator of the heritability, so the trained trait 1 is the trait 1 of `PCH` with ridge s^2 / v (v: the mean variance of the measurements).

Options: `n_iter` steps per trait (default 10000); `optimizer`, `'adam'` (default) or `'rmsprop'`; `learning_rate` (default 1e-3); `noise_level` for data augmentation (default 0.1), one level for all traits or one per trait: the maximum of the uniform noise added to the measurements at each step, or its standard deviation with `noise='normal'`; `clip` to clip the standardized traits in the loss to [-clip, clip] (default `None`: no clipping, so the loss is the plain heritability; the paper clipped sorghum at 2); `model_file` to save the model after each trait; and `device` (e.g. `"cuda"`) to train on a GPU.
`estimator` chooses the heritability that is optimized and reported: `'anova'` (default, `anova_heritability`) or `'henderson3'` (`Henderson3`, with `subgroups`, e.g. plots of one family).
`'adam'` uses momentum 0.99 and decays the learning rate to 0 on a cosine schedule over the `n_iter` steps of each trait.
`'rmsprop'` is the optimizer of the paper (constant learning rate). It converges slowly when the measurements are strongly correlated: the heritability does not change with the scale of a trait, so its steps shrink relative to the weights as the weights grow.
`ConvModel` is a convolutional alternative to `LinearModel` for spectra, and `ImageConvModel(n_channels, image_size)` one for multispectral images (input n x channels x height x width; the paper's Miscanthus model).

`train_batch` trains one trait in each of many models at once, each with its own training individuals, noise level and earlier traits.
`train_models` trains several `TraitModels` at once on it, each with its own `train_test` and noise levels (e.g. the outer folds of a dataset, or one model per noise level); `train` is `train_models` with one model.
The copies share each noise draw, scaled by their own level, and each copy gets only its own gradient. `max_copies` trains at most that many copies at once, for models too large to train all copies together on a GPU.
For `LinearModel` it is fast, because (X + noise) w = X w + noise w needs no noisy copy of X per model, and with normal noise each model's noise w is drawn directly (normal with standard deviation |w| per individual).
On a GPU, one step is recorded as a CUDA graph and replayed, which removes the cost of launching its many small kernels.
On an RTX 3080, one sorghum-sized linear model trains at about 2,500 steps per second, and 400 at about 210,000 model-steps per second.
Other models are trained with `torch.func.vmap`, which gives no speedup for `ConvModel`.

### Choosing the noise level

`select_noise_level` chooses one noise level for all traits from the data. For each level, it trains all traits in order on part of the groups and scores their heritability on the held-out groups.
Each level is the standard deviation of normal noise as a fraction of `noise_scale(X)`, the root mean variance of the measurements, so the same levels apply to data on any scale.

```python
from h2opt.selection import noise_scale, select_noise_level

is_train = train_test == 0
level, scores, traits = select_noise_level(lambda: LinearModel(X.shape[1]), X, groups, environment,
                                           n_traits, subsets=[is_train], n_iter=10000)
train(model, X, groups, environment, train_test, n_traits=n_traits,
      noise_level=level[0] * noise_scale(X[is_train]), noise='normal')
```

Within each subset of the individuals (`subsets`, e.g. the training individuals of several outer folds, all chosen in one batch), the groups are split into 5 folds, and each fold is the held-out groups of one validation split (`n_folds`; `n_splits` uses only the first folds).
The levels are `NOISE_LEVELS` (0.001 to 10), the square roots of the PCH ridges `RIDGES`, so that level s has the penalty of ridge s^2. The chosen level has the best held-out heritability averaged over the traits and splits, the same rule as `PCH.tune`.
All subsets, splits and levels of a trait are trained together by `train_batch`.
Given a list of measurement sets of the same individuals (e.g. one per date), it trains and scores each set and chooses one level for all of them by their mean score, as `PCH.tune` chooses one ridge.
`select_noise_level` also returns the scores of each trait and the trained validation traits (before decorrelation) for each subset, trait, split and level. The traits of a level do not depend on the other levels, so another rule over the levels can be examined without training again.

## Baselines

`h2opt.baselines` has linear baselines with a common interface: `fit(X, groups, environment)` on the training individuals, then `transform(X)`.
Their traits are centered and made uncorrelated on the training individuals (`Decorrelation`), like H2-opt's.

- `PCA(n_traits)`: principal components of the measurements.
- `GeneticPCA(n_traits)`: principal components of the ANOVA estimate of the genetic covariance.
- `LDA(n_traits)`: linear discriminant analysis of the groups (unregularized).
- `PCH(n_traits, ridge, estimator='anova')`: principal components of heritability, maximizing the ANOVA heritability above (or, with `estimator='henderson3'` and `fit(X, groups, environment, subgroups)`, Henderson's Method III) with ridge regularization.
  Both estimators are ratios of quadratic forms in the trait, so the solution is exact. PCH works in the span of the training data, so the measurements may far outnumber the individuals (e.g. image pixels).
  `PCH.tune` chooses the ridge from `RIDGES` (1e-6 to 100) by the highest mean held-out heritability, with the same estimator; given a list of measurement sets of the same individuals (e.g. one per date), it chooses one ridge for all of them by their mean score.
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
The pretrained autoencoder in `data/examples/autoencoder.pt` has 5 latent dimensions.

Example with the latent traits simulated with simplePHENOTYPES in the paper:

```python
import numpy as np
from h2opt import load_npz, encode_latent, latent_scale, AutoEncoder

autoencoder = AutoEncoder.load('data/examples/autoencoder.pt')
reference = np.concatenate((load_npz('data/examples/X_file1.npz'), load_npz('data/examples/X_file2.npz')))
latent = load_npz('data/examples/simulatedLatentTraits.npz')
latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)

X_simulated = encode_latent(latent, autoencoder, latent_scale(autoencoder, reference))
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
