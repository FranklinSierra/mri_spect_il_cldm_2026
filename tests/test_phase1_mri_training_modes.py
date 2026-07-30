"""Phase-1 MRI training mode config defaults."""

import config


def test_default_training_mode_is_recon():
    assert config.MRI_TRAINING_MODE == "recon"


def test_contrastive_config_defaults():
    assert config.SUPCON_LAMBDA == 20.0
    assert config.XMODAL_LAMBDA == 20.0
    assert config.CONTRASTIVE_TEMP == 0.5
    assert config.CONTRASTIVE_FEATURES == "z18k"
    assert config.batch_size == 8
