"""Registry resolution for calib_roi3ch_uni_rois_reproc_film_ms."""

from scripts.run_ablation import load_registry, resolve_env


def test_calib_roi3ch_uni_rois_reproc_film_ms():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_film_ms")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc"
    assert entry["short_label"] == "rois_reproc_film_ms"
    assert entry["split_profile"] == "data_rois"
    assert "init_checkpoint" not in entry
    assert env["ILCLDM_FILM_MODE"] == "spatial_ms"
    assert env["ILCLDM_USE_FILM"] == "0"
    assert env["ILCLDM_LABEL_DROPOUT_PROB"] == "0"
    # No EMA override: the entry now inherits the config defaults (warmup 0.9 -> 0.995).
    # The legacy 0.9999 froze the evaluated weights at the step_start snapshot.
    assert "ILCLDM_EMA_BETA" not in env
    assert "ILCLDM_EMA_BETA_START" not in env
    assert "ILCLDM_EMA_WARMUP_STEPS" not in env
    assert env["ILCLDM_PARENT_EXPERIMENT"] == "calib_roi3ch_uni_rois_reproc"
    assert env["ILCLDM_EXP"] == "exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_film_ms/"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_reproc/"
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "frozen"


def test_calib_roi3ch_uni_rois_reproc_film_ms_ft():
    registry = load_registry()
    entry, env = resolve_env(registry, "calib_roi3ch_uni_rois_reproc_film_ms_ft")
    assert entry["parent"] == "calib_roi3ch_uni_rois_reproc_film_ms"
    assert entry["base"] == "base_calib_roi3ch_uni_rois_reproc"
    assert entry["short_label"] == "rois_reproc_film_ms_ft"
    assert entry["split_profile"] == "data_rois"
    assert "init_checkpoint" not in entry
    assert env["ILCLDM_MRI_ENCODER_MODE"] == "finetune"
    assert env["ILCLDM_FILM_MODE"] == "spatial_ms"
    assert env["ILCLDM_USE_FILM"] == "0"
    assert env["ILCLDM_LABEL_DROPOUT_PROB"] == "0"
    assert env["ILCLDM_PARENT_EXPERIMENT"] == "calib_roi3ch_uni_rois_reproc_film_ms"
    assert env["ILCLDM_EXP"] == "exp_nm_s2_finetune_calib_roi3ch_uni_rois_reproc_film_ms/"
    assert env["ILCLDM_LATENT_ABETA"] == "data_rois/latent_Abeta_reproc/"
    assert env["ILCLDM_MRI_ENCODER_CHECKPOINT"].endswith(
        "exp_nm_s1_mri_rois_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar"
    )
