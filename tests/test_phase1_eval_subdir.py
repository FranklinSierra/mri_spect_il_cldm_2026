"""Tests for alternate embedding/analysis output dirs (e.g. z18k eval on h64 run)."""

from types import SimpleNamespace

from code.run_phase1_mri_experiment import paths_for, resolve_eval_feature_mode


def test_paths_for_custom_analysis_and_embeddings():
    path_map = paths_for(
        "exp_nm_s1_mri_rois_xmodal_h64",
        embeddings_name="embeddings_z18k",
        analysis_name="embedding_analysis_z18k",
    )
    assert path_map["embeddings"].name == "embeddings_z18k"
    assert path_map["analysis"].name == "embedding_analysis_z18k"
    assert path_map["classification"].as_posix().endswith(
        "embedding_analysis_z18k/classification"
    )
    assert path_map["metadata"].name == "embeddings_z18k_metadata.csv"
    # Checkpoint stays shared with the original training run.
    assert path_map["checkpoint"].as_posix().endswith("checkpoints/MRI_AAE.pth.tar")


def test_resolve_eval_feature_mode_override():
    args = SimpleNamespace(
        eval_features="z18k",
        contrastive_features="h64",
        training_mode="xmodal",
        skip_train=True,
    )
    assert resolve_eval_feature_mode(args) == "z18k"


def test_resolve_eval_feature_mode_skip_train_uses_contrastive_flag():
    args = SimpleNamespace(
        eval_features=None,
        contrastive_features="z18k",
        training_mode="recon",
        skip_train=True,
    )
    assert resolve_eval_feature_mode(args) == "z18k"
