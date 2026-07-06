"""Registry resolution for roi3ch wave experiments."""

from pathlib import Path

from scripts.run_ablation import load_registry, resolve_env

ROOT = Path(__file__).resolve().parents[1]


def test_roi3ch_nhy_split_and_checkpoints():
    registry = load_registry()
    entry, env = resolve_env(registry, "roi3ch_nhy")
    assert entry["split_profile"] == "nhy_v1"
    assert "init_checkpoint" not in entry
    assert "parent" not in entry
    assert env["ILCLDM_TRAIN_FILE"] == "data_info/splits/nhy_v1/train.txt"
    assert env["ILCLDM_LATENT_ABETA"] == "data/latent_Abeta_nhy_v1/"
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_nhy_v1/AAE.pth.tar"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"


def test_abl_hight05_nhy_scratch():
    registry = load_registry()
    entry, env = resolve_env(registry, "abl_hight05_nhy")
    assert entry["split_profile"] == "nhy_v1"
    assert entry["base"] == "base_frozen_nhy"
    assert "init_checkpoint" not in entry
    assert "parent" not in entry
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0.5"
    assert env["ILCLDM_TRAIN_FILE"] == "data_info/splits/nhy_v1/train.txt"
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_nhy_v1/AAE.pth.tar"
    assert "ILCLDM_UNET_ROI_CHANNEL" not in env


def test_h05_roi3ch_nhy_warm_start():
    registry = load_registry()
    entry, env = resolve_env(registry, "h05_roi3ch_nhy")
    assert entry["split_profile"] == "nhy_v1"
    assert entry["parent"] == "abl_hight05_nhy"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"
    assert env["ILCLDM_UNET_INIT_CHECKPOINT"].endswith("abl_hight05_nhy/Unet.pth.tar")
    assert env["ILCLDM_TRAIN_FILE"] == "data_info/splits/nhy_v1/train.txt"
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_nhy_v1/AAE.pth.tar"


def test_roi3ch_nolabel_warm_start():
    registry = load_registry()
    entry, env = resolve_env(registry, "roi3ch_nolabel")
    assert entry["parent"] == "h05_roi3ch"
    assert env["ILCLDM_USE_TRAIN_LABEL"] == "0"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"
    assert env["ILCLDM_UNET_INIT_CHECKPOINT"].endswith("h05_roi3ch/Unet.pth.tar")


def test_roi3ch_ht08_high_t():
    registry = load_registry()
    _, env = resolve_env(registry, "roi3ch_ht08")
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0.8"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"


def test_roi3ch_cls_flags():
    registry = load_registry()
    entry, env = resolve_env(registry, "roi3ch_cls")
    assert entry["early_stop"] == "cls"
    assert env["ILCLDM_CLS_LOSS"] == "1"
    assert env["ILCLDM_EARLY_STOP_CLS_METRIC"] == "1"
    assert env["ILCLDM_USE_TRAIN_LABEL"] == "0"


def test_h05_calib_roi3ch_warm_start():
    registry = load_registry()
    entry, env = resolve_env(registry, "h05_calib_roi3ch")
    assert entry["parent"] == "calib_roi3ch_uni"
    assert env["ILCLDM_LATENT_CALIBRATION"] == "1"
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0.5"
    assert env["ILCLDM_HIGH_T_THRESHOLD"] == "700"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"
    assert env["ILCLDM_UNET_INIT_CHECKPOINT"].endswith("calib_roi3ch_uni/Unet.pth.tar")
