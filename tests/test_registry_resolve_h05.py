import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import load_registry, resolve_env


def test_h05_ht08_resolves_base_and_warm_start():
    registry = load_registry()
    entry, env = resolve_env(registry, "h05_ht08")

    assert entry["parent"] == "abl_hight05"
    assert entry["init_checkpoint"] == "result/exp_nm_s2_frozen_abl_hight05/Unet.pth.tar"
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0.8"
    assert env["ILCLDM_HIGH_T_THRESHOLD"] == "700"
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == entry["overrides"]["ILCLDM_HIGH_T_SAMPLE_PROB"]
    assert env["ILCLDM_REGISTRY_KEY"] == "h05_ht08"
    assert env["ILCLDM_PARENT_EXPERIMENT"] == "abl_hight05"
    assert env["ILCLDM_UNET_INIT_CHECKPOINT"] == entry["init_checkpoint"]
    assert env["ILCLDM_EXP"] == "exp_nm_s2_frozen_h05_ht08/"


def test_h05_roi3ch_enables_unet_roi_channel():
    registry = load_registry()
    entry, env = resolve_env(registry, "h05_roi3ch")

    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"
    assert env["ILCLDM_REGISTRY_KEY"] == "h05_roi3ch"
    assert env["ILCLDM_UNET_INIT_CHECKPOINT"].endswith("abl_hight05/Unet.pth.tar")


def test_h05_nolabel_disables_train_label():
    registry = load_registry()
    _, env = resolve_env(registry, "h05_nolabel")
    assert env["ILCLDM_USE_TRAIN_LABEL"] == "0"


def test_h05_aux5_sets_aux_every_n():
    registry = load_registry()
    _, env = resolve_env(registry, "h05_aux5")
    assert env["ILCLDM_AUX_IMAGE_LOSS"] == "1"
    assert env["ILCLDM_AUX_IMAGE_EVERY_N"] == "5"
    assert env["ILCLDM_AUX_IMAGE_LAMBDA"] == "0.25"
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0.5"
