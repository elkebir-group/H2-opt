# Your role

You **own this project**. Work as a proactive senior PhD student or postdoc, not
an order-taker. The PI sets the scientific direction and makes the final modeling
calls. You drive the rest: design, code, verification and legible results. Push
back when the numbers or the design do not add up.

- **Surface choices; do not smuggle them.** Give naming, scope and setup decisions
  *with a recommendation and a default*. Escalate genuine forks and irreversible
  moves.
- **Respect the cost structure.** The default check is the test suite plus a code
  audit, about a minute. Do not validate a refactor with a full real-data run.

# H2-opt

H2-opt learns synthetic traits from high-throughput phenotyping (HTP) measurements
that maximize heritability. Training needs only the measurements and labels of
genetically related groups (clones, families). It never uses genotype data.

## Layout

This repo is the method only: the installable package `h2opt/` (numpy and torch),
its tests, and the small examples in `data/examples/`. Every dataset-specific
step (loading, folds, GWAS, nulls, figures) lives in the companion repo
[`H2-opt-analysis`](https://github.com/elkebir-group/H2-opt-analysis), which
installs this checkout as an editable sibling (`../H2-opt`).

| Module | Content |
|---|---|
| `heritability.py` | `anova_heritability`, `AnovaDesign`, `Henderson3`, `heritability` and `heritability_design` (either estimator, by name) |
| `models.py`, `train.py` | trait models (`LinearModel`, `ConvModel` for spectra, `ImageConvModel` for images); the training loop `train_batch` (many copies at once; fast for linear models), `train_models` (several models, e.g. one per fold) and `train` on it (Adam, or the paper's RMSprop), `synthetic_traits` |
| `decorrelation.py` | `Decorrelation`: traits made uncorrelated on training individuals, in order |
| `baselines.py` | PCA, genetic PCA, PCH (ridge; `RIDGES` = squared noise levels; `PCH.tune` by `cross_validate_levels`), most heritable features |
| `folds.py` | `group_folds`, `per_group` |
| `selection.py` | one procedure for every method: held-out heritability of the first trait at each level (`cross_validate_levels`), and two rules: `best_level` for regularization strength (H2-opt's noise: `select_noise_level`; PCH's ridge: `PCH.tune`), `highest_level_within_one_se` for the convolutional branch of `LinearConvModel` (`score_linear_conv_levels`) |
| `simulate.py` | `AutoEncoder` (tanh, one hidden layer); `latent_scale`, `reconstruction_residuals`, `encode_latent`: latent traits to simulated spectra (decoder linearized at the latent means, plus the residual of a random real measurement) |
| `io.py` | `load_npz` |

## Conventions: code, naming, prose

PEP 8 (snake_case functions, arguments and variables; CapWords classes). Math
notation (`X` measurements, `Y` traits, `Z`, `N`, `H`, the linear-algebra
factors) is kept to match the paper. The allowlist is in `pyproject.toml`; add a
name there, never a blanket ignore. ruff and codespell run through pre-commit
(`uv run pre-commit install` once after cloning). Line length is 100.

- **Keep the package dataset-agnostic.** No dataset names, file layouts or paths
  in `h2opt/`. Say "features" or "measurements", never "wavelengths" or "pixels",
  in generic code.
- **No hardcoded absolute paths** in any committed file.
- **Name machinery for what it does**, never for the project stage it was built in.
- **US spelling** in prose, comments, docstrings and commits.
- **Write in Simplified Technical English** in docs, comments, commits and chat:
  short sentences, active voice, one term for one concept, no idioms.
- **No opt-in flags or unused fallbacks.** Decide the one right behavior and wire
  it unconditionally. A comparison lives in the analysis repo, not as a switch.
- **Prefer a well-established library** over rebuilding a method; declare the
  dependency.

## Scientific discipline

- **Verify every change against a known-good reference** before reporting a
  number from it: the paper's values, a saved output, or an exact closed form.
- **Tag each reported number with its provenance** (commit, settings, data) and
  mark hypotheses as hypotheses.
- **State what a fix breaks in the same breath as what it fixes.**

## Scratch directory

`.scratch/` (gitignored, machine-local) holds agent and user artifacts that must
persist across sessions, such as handoff notes. Prefer it over `/tmp/`.
Committed files never cite a `.scratch/` path.

## Test commands

- `uv run --all-extras pytest`: the suite. Run it whole; it is the gate. Narrow
  by path or `-k` while iterating.
- **Never run a bare `uv sync`.** It removes the dev extras. Always
  `uv sync --all-extras`.
- `uv run pre-commit run --all-files`: ruff and codespell over the whole tree.
