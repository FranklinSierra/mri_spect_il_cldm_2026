# IL-CLDM Dev Container

This project can run inside a VS Code Dev Container to isolate Python,
PyTorch, CUDA libraries, and project dependencies from other users and
other paper repositories on the shared machine.

## Open

1. Open this repository in VS Code through Remote SSH.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for the image build and `postCreateCommand` check to finish.

Volumes live under `data/` inside the repo (gitignored). By default the
dev container uses the workspace checkout; optional host bind via
`HOST_DATA_DIR` (see `.env.example` and `start_devcontainer.sh`).

## Smoke Test

```bash
python smoke_test_fsierra.py
```

## Short Run

```bash
ILCLDM_EPOCHS=1 ILCLDM_NOISE_STEPS=10 ./run_fsierra.sh
```

`run_fsierra.sh` is only for the original fsierra symlink workflow
(`prepare_fsierra_data.py`). If you are using local `data/` replaced with
`temp_data/non_masked`, use `scripts/run_experiment.sh` instead.

## Full Run

```bash
./run_fsierra.sh
```

By default the container exposes all GPUs, while `run_fsierra.sh` selects
GPU `6` using `CUDA_VISIBLE_DEVICES=6`.

## Remote Explorer Fallback

If VS Code fails while creating the Dev Container, start it manually:

```bash
cd /path/to/IL-CLDM
cp .env.example .env   # optional: set HOST_DATA_DIR, CUDA_VISIBLE_DEVICES
./start_devcontainer.sh
```

Then open `Remote Explorer > Dev Containers`, find `il-cldm-dev`, and
attach to the running container.

The same container also mounts CoCoLIT at `/workspace/CoCoLIT` (sibling
project under `IL-CLDM/CoCoLIT/` on disk). See
[CoCoLIT/RUNBOOK_COCOLIT.md](CoCoLIT/RUNBOOK_COCOLIT.md) for the baseline
pipeline. To run CoCoLIT in its own container:

```bash
cd IL-CLDM/CoCoLIT
./start_devcontainer.sh
```

## Data layout (`data/`)

Image volumes live under `data/whole_MRI` and `data/whole_Abeta`. On every
load through `dataset.nifti_to_numpy`, **per-volume z-score** is applied
(mean 0, std 1 over all voxels). Latent NIfTI under `latent_Abeta/` and
`latent_MRI/` are loaded without z-score.

After replacing raw volumes (e.g. from `temp_data/non_masked`), re-run
encoding to refresh `data/latent_Abeta/`.

Internal audit reports can be written under `docs/local/` (gitignored).
Legacy paths: `docs/reports/fsierra_source_audit.json`.

## Visualization (z-score volumes)

```bash
python code/visualize_pair_example.py
```

Use `viz_pairs.display_range(..., symmetric=True)` for MRI. Do **not** scale
with `voxels > 0` only — z-score MRI has negative values in much of the tissue.

## Peak-aware training (optional env vars)

Stage 1 AAE and stage 2 auxiliary decode loss (defaults on):

```bash
export ILCLDM_USE_WEIGHTED_RECON=1    # weight voxels with target z > 1
export ILCLDM_USE_TAIL_RECON=1        # extra L1 on top 5% intensities
export ILCLDM_WEIGHTED_ALPHA=3.0
export ILCLDM_AUX_IMAGE_LOSS=1        # decode(x0) vs SPECT every N steps
export ILCLDM_AUX_IMAGE_EVERY_N=10
export ILCLDM_AUX_IMAGE_LAMBDA=0.25
```

## Recommended experiment order (non_masked in `data/`)

Copy/paste commands, GPU, and smoke vs full training: **[RUNBOOK.md](RUNBOOK.md)**.

### 0) Preflight

```bash
scripts/run_experiment.sh preflight
```

### 1) Stage 1 SPECT AAE + latent encoding (`latent_Abeta`)

```bash
scripts/run_experiment.sh s1_spect exp_nm_s1_spect
```

### 2) Stage 1 MRI AAE replication + embedding analysis

```bash
scripts/run_experiment.sh s1_mri exp_nm_s1_mri
```

### 3) Stage 2 baseline diffusion (raw MRI conditioner)

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
scripts/run_experiment.sh s2_baseline exp_nm_s2_baseline
```

### 4) Stage 2 with pretrained MRI encoder frozen (only UNet trained)

```bash
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar
scripts/run_experiment.sh s2_mri_frozen exp_nm_s2_mri_frozen
```

### 4b) Stage 2 frozen v2 (latent loss, high-t sampling, ROI aux, latent early stop)

Uses basal ganglia mask at `data/mask_basal_ganglia.nii.gz` (same crop as volumes). DDIM sampling stays at 100 steps.

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar
scripts/run_experiment.sh s2_mri_frozen_v2 exp_nm_s2_mri_frozen_v2
```

Key env vars (set by the script): `ILCLDM_LATENT_LOSS=1`, `ILCLDM_LATENT_LOSS_LAMBDA=0.5`, `ILCLDM_AUX_IMAGE_EVERY_N=1`, `ILCLDM_AUX_IMAGE_LAMBDA=0.5`, `ILCLDM_HIGH_T_SAMPLE_PROB=0.5`, `ILCLDM_HIGH_T_THRESHOLD=700`, `ILCLDM_USE_ROI_LOSS=1`, `ILCLDM_EARLY_STOP_LATENT_METRIC=1`.

```bash
python inference.py --exp exp_nm_s2_mri_frozen_v2 --run-id nm_stage2_v2 --save-synthetic-spect
```

### 5) Stage 2 with pretrained MRI encoder fine-tuned

```bash
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar
export ILCLDM_MRI_ENCODER_FINETUNE_LR_SCALE=0.25
scripts/run_experiment.sh s2_mri_finetune exp_nm_s2_mri_finetune
```

### Inference/evaluation

```bash
python inference.py --exp exp_nm_s2_baseline --save-synthetic-spect
python inference.py --exp exp_nm_s2_mri_frozen --save-synthetic-spect
python inference.py --exp exp_nm_s2_mri_finetune --save-synthetic-spect
```

## Experiment naming keys

- `ILCLDM_EXP` controls output folder names (`result/<exp>/...`).
- `ILCLDM_AAE_CHECKPOINT` optional path to a frozen SPECT AAE for stage 2 (default: `result/<exp>/AAE.pth.tar`).
- `ILCLDM_MRI_ENCODER_MODE=raw|frozen|finetune` controls stage-2 conditioning mode.
- Backward compatibility:
  - `ILCLDM_FREEZE_MRI_ENCODER=1` maps to frozen mode if `ILCLDM_MRI_ENCODER_MODE` is not set.
