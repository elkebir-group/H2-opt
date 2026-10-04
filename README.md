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

Options: `n_iter` steps per trait (default 10000); `optimizer`, `'adam'` (default) or `'rmsprop'`; `learning_rate` (default 1e-3); `noise_level` for data augmentation (default 0.1): the maximum of the uniform noise added to the measurements at each step, or its standard deviation with `noise='normal'`; `clip` for the standardized traits in the loss (default 2, `None` for no clipping); `model_file` to save the model every `save_every` steps (default 1000) and after each trait; and `device` (e.g. `"cuda"`) to train on a GPU.
`'adam'` uses momentum 0.99 and decays the learning rate to 0 on a cosine schedule over the `n_iter` steps of each trait, so a model saved before the end of a trait is not fully trained.
`'rmsprop'` is the optimizer of the paper (constant learning rate). It converges slowly when the measurements are strongly correlated: the heritability does not change with the scale of a trait, so its steps shrink relative to the weights as the weights grow.
`ConvModel` is a convolutional alternative to `LinearModel` for spectra.

### Choosing the noise level

`select_noise_level` chooses the noise level from the data. For each level, it trains the first trait on part of the groups and scores its heritability on the held-out groups.
Each level is the standard deviation of normal noise as a fraction of `noise_scale(X)`, the root mean variance of the measurements, so the same levels apply to data on any scale.

```python
from h2opt.selection import noise_scale, select_noise_level

level, scores = select_noise_level(lambda: TraitModels(1, LinearModel, X.shape[1]),
                                   X, groups, environment, n_iter=10000)
train(model, X, groups, environment, train_test, n_traits=n_traits,
      noise_level=level * noise_scale(X), noise='normal')
```

The groups are split into 5 folds, and folds 0 and 1 are the held-out groups of two validation splits (`n_splits`, `n_folds`).
The levels are `NOISE_LEVELS` (0 to 0.3), and the chosen level has the highest mean held-out heritability over the splits.

## Baselines

`h2opt.baselines` has linear baselines with a common interface: `fit(X, groups, environment)` on the training individuals, then `transform(X)`.
Their traits are centered and made uncorrelated on the training individuals (`Decorrelation`), like H2-opt's.

- `PCA(n_traits)`: principal components of the measurements.
- `GeneticPCA(n_traits)`: principal components of the ANOVA estimate of the genetic covariance.
- `LDA(n_traits)`: linear discriminant analysis of the groups (unregularized).
- `PCH(n_traits, ridge)`: principal components of heritability, maximizing the ANOVA heritability above with ridge regularization.
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
