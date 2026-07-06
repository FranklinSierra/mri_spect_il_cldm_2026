"""Config split path overrides via environment."""

import importlib
import os


def test_config_split_defaults(monkeypatch):
    for key in (
        "ILCLDM_TRAIN_FILE",
        "ILCLDM_TEST_FILE",
        "ILCLDM_VALIDATION_FILE",
        "ILCLDM_LATENT_ABETA",
    ):
        monkeypatch.delenv(key, raising=False)
    import config

    importlib.reload(config)
    assert config.train == "data_info/train.txt"
    assert config.test == "data_info/test.txt"
    assert config.latent_Abeta == "./data/latent_Abeta/"


def test_config_split_overrides(monkeypatch):
    monkeypatch.setenv("ILCLDM_TRAIN_FILE", "data_info/splits/nhy_v1/train.txt")
    monkeypatch.setenv("ILCLDM_TEST_FILE", "data_info/splits/nhy_v1/test.txt")
    monkeypatch.setenv("ILCLDM_VALIDATION_FILE", "data_info/splits/nhy_v1/validation.txt")
    monkeypatch.setenv("ILCLDM_LATENT_ABETA", "data/latent_Abeta_nhy_v1/")
    import config

    importlib.reload(config)
    assert config.train.endswith("nhy_v1/train.txt")
    assert config.latent_Abeta == "data/latent_Abeta_nhy_v1/"
