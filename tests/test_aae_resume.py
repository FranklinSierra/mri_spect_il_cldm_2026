from pathlib import Path

from utils import resolve_aae_resume


def test_resolve_aae_resume_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ILCLDM_AAE_RESUME", raising=False)
    start, average, should_load = resolve_aae_resume(
        tmp_path / "AAE.pth.tar", tmp_path, "stage1_abeta_autoencoder"
    )
    assert (start, average, should_load) == (0, 0.0, False)


def test_resolve_aae_resume_from_validation_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("ILCLDM_AAE_RESUME", "1")
    ckpt = tmp_path / "AAE.pth.tar"
    ckpt.write_bytes(b"ckpt")
    (tmp_path / "stage1_abeta_autoencoder_validation.csv").write_text(
        "Epoch,PSNR,SSIM\n65,40.19,0.977\n70,40.225,0.9808\n"
    )
    start, average, should_load = resolve_aae_resume(
        ckpt, tmp_path, "stage1_abeta_autoencoder"
    )
    assert should_load is True
    assert start == 70
    assert abs(average - (40.225 + 0.9808 * 10.0)) < 1e-9
