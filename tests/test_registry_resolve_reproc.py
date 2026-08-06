"""Registry resolution for reproc / reproc_v2load calib_roi3ch_uni pairs."""

from scripts.run_ablation import load_registry, resolve_env


def test_calib_roi3ch_uni_reproc_legacy_load():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_reproc")
    assert entry["base"] == "base_calib_roi3ch_uni_reproc"
    assert entry["exp"] == "exp_nm_s2_frozen_calib_roi3ch_uni_reproc"
    assert "init_checkpoint" not in entry
    assert env["ILCLDM_IMAGE_VOLUME_NORM"] == "legacy"
    assert env["ILCLDM_LATENT_CALIBRATION"] == "1"
    assert env["ILCLDM_HIGH_T_SAMPLE_PROB"] == "0"
    assert env["ILCLDM_UNET_ROI_CHANNEL"] == "1"
    assert env["ILCLDM_LATENT_ABETA"] == "data/latent_Abeta_reproc/"
    assert env["ILCLDM_LATENT_STATS_PATH"] == "data_info/latent_stats_latent_Abeta_reproc.json"
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_reproc/AAE.pth.tar"
    assert env["ILCLDM_MRI_ENCODER_CHECKPOINT"].endswith(
        "exp_nm_s1_mri_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar"
    )
    assert env["ILCLDM_EXP"] == "exp_nm_s2_frozen_calib_roi3ch_uni_reproc/"


def test_calib_roi3ch_uni_reproc_v2load():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_reproc_v2load")
    assert entry["base"] == "base_calib_roi3ch_uni_reproc_v2load"
    assert env["ILCLDM_IMAGE_VOLUME_NORM"] == "v2load"
    assert env["ILCLDM_LATENT_ABETA"] == "data/latent_Abeta_reproc_v2load/"
    assert env["ILCLDM_LATENT_STATS_PATH"] == (
        "data_info/latent_stats_latent_Abeta_reproc_v2load.json"
    )
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_reproc_v2load/AAE.pth.tar"
    assert "rois" not in env.get("ILCLDM_WHOLE_MRI", "data/whole_MRI")


def test_calib_roi3ch_uni_rois_reproc_legacy_load():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc")
    assert entry["split_profile"] == "data_rois"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc"
    assert env["ILCLDM_IMAGE_VOLUME_NORM"] == "legacy"
    assert env["ILCLDM_SKIP_CROP"] == "1"
    assert env["ILCLDM_WHOLE_MRI"] == "data_rois/whole_MRI"
    assert env["ILCLDM_WHOLE_ABETA"] == "data_rois/whole_Abeta"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_reproc/"
    assert env["ILCLDM_LATENT_STATS_PATH"] == (
        "data_info/latent_stats_latent_Abeta_rois_reproc.json"
    )
    assert env["ILCLDM_AAE_CHECKPOINT"] == "result/exp_nm_s1_spect_rois_reproc/AAE.pth.tar"
    assert env["ILCLDM_ROI_MASK_PATH"] == "data_rois/roi_mask_unet.nii.gz"


def test_calib_roi3ch_uni_rois_reproc_v2load():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_v2load")
    assert entry["split_profile"] == "data_rois"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc_v2load"
    assert env["ILCLDM_IMAGE_VOLUME_NORM"] == "v2load"
    assert env["ILCLDM_SKIP_CROP"] == "1"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_reproc_v2load/"
    assert env["ILCLDM_LATENT_STATS_PATH"] == (
        "data_info/latent_stats_latent_Abeta_rois_reproc_v2load.json"
    )
    assert env["ILCLDM_AAE_CHECKPOINT"] == (
        "result/exp_nm_s1_spect_rois_reproc_v2load/AAE.pth.tar"
    )
    assert env["ILCLDM_MRI_ENCODER_CHECKPOINT"].endswith(
        "exp_nm_s1_mri_rois_reproc_v2load/phase1_mri/checkpoints/MRI_AAE.pth.tar"
    )
