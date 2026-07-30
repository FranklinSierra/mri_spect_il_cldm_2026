"""Loss functions for IL-CLDM."""

from losses.contrastive import (
    ContrastiveProjector,
    CrossModalInfoNCE,
    SupervisedContrastiveLoss,
    flatten_feature_volume,
    l2_normalize,
    pool_latent_volume,
)
from losses.spect import (
    combined_spect_weights,
    spect_reconstruction_loss,
    tail_l1_loss,
    uptake_weights,
    weighted_l1_loss,
)

__all__ = [
    "ContrastiveProjector",
    "CrossModalInfoNCE",
    "SupervisedContrastiveLoss",
    "combined_spect_weights",
    "flatten_feature_volume",
    "l2_normalize",
    "pool_latent_volume",
    "spect_reconstruction_loss",
    "tail_l1_loss",
    "uptake_weights",
    "weighted_l1_loss",
]
