"""H2-opt: self-supervised discovery of heritable traits in high-throughput phenotyping data."""

from . import baselines, folds, selection
from .decorrelation import Decorrelation
from .heritability import (
    ESTIMATORS,
    AnovaDesign,
    Henderson3,
    anova_heritability,
    heritability,
    heritability_design,
)
from .io import load_npz
from .models import (
    ConvModel,
    ImageConvModel,
    LinearConvModel,
    LinearModel,
    PositionImageConvModel,
    TraitModels,
)
from .simulate import AutoEncoder, encode_latent, latent_scale, reconstruction_residuals
from .train import synthetic_traits, train, train_batch, train_linear_conv_models, train_models

__all__ = [
    "ESTIMATORS",
    "AnovaDesign",
    "AutoEncoder",
    "ConvModel",
    "Decorrelation",
    "Henderson3",
    "ImageConvModel",
    "PositionImageConvModel",
    "LinearConvModel",
    "LinearModel",
    "TraitModels",
    "anova_heritability",
    "baselines",
    "folds",
    "selection",
    "encode_latent",
    "heritability",
    "heritability_design",
    "latent_scale",
    "load_npz",
    "reconstruction_residuals",
    "synthetic_traits",
    "train",
    "train_batch",
    "train_linear_conv_models",
    "train_models",
]
