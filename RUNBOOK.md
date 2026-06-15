# RUNBOOK — IL-CLDM con `data/` (non_masked)

Guía corta para copiar/pegar. Datos en `data/whole_MRI` y `data/whole_Abeta` (184 sujetos). Normalización **z-score por volumen** al cargar (`dataset.nifti_to_numpy`). **No** ejecutar `prepare_fsierra_data.py` ni `./run_fsierra.sh` con esta cohorte.

---

## 0. Entorno común (ajusta GPU y duración)

Desde la raíz del repo (directorio con `config.py`):

```bash
cd /path/to/IL-CLDM

# GPU (ejemplo: GPU 6 en el servidor compartido)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# Hiperparámetros globales (ver config.py)
export ILCLDM_LR=1e-4
export ILCLDM_BATCH_SIZE=2
export ILCLDM_VALIDATION_INTERVAL=5
export ILCLDM_EARLY_STOPPING_PATIENCE=10

# Pérdidas orientadas a picos SPECT (recomendado con z-score)
export ILCLDM_USE_WEIGHTED_RECON=1
export ILCLDM_USE_TAIL_RECON=1
export ILCLDM_WEIGHTED_ALPHA=3.0
export ILCLDM_WEIGHTED_Z_THRESH=1.0
export ILCLDM_TAIL_FRACTION=0.05
export ILCLDM_TAIL_BETA=1.0

# Etapa 2: pérdida auxiliar en imagen SPECT
export ILCLDM_AUX_IMAGE_LOSS=1
export ILCLDM_AUX_IMAGE_EVERY_N=10
export ILCLDM_AUX_IMAGE_LAMBDA=0.25
```

### Perfil rápido (smoke / debug)

```bash
export ILCLDM_EPOCHS=2
export ILCLDM_NOISE_STEPS=50
export ILCLDM_SAMPLING_STEPS=20
export ILCLDM_AUX_IMAGE_EVERY_N=1
```

### Perfil entrenamiento completo

```bash
export ILCLDM_EPOCHS=1000
export ILCLDM_NOISE_STEPS=1000
export ILCLDM_SAMPLING_STEPS=100
```

(Omitir esas variables usa los mismos valores por defecto en `config.py`.)

---

## 1. Preflight (obligatorio antes de entrenar)

Sincroniza metadatos si cambiaste `data/`:

```bash
python3 scripts/sync_data_info_to_data.py
```

Comprueba IDs, splits y smoke test:

```bash
scripts/run_experiment.sh preflight
```

Esperado: 184 IDs, 0 faltantes; `latent_Abeta` vacío **antes** de E1; smoke test OK.

---

## 2. E1 — Fase 1 SPECT (AAE + latentes)

Entrena el autoencoder SPECT y rellena `data/latent_Abeta/`.

```bash
scripts/run_experiment.sh s1_spect exp_nm_s1_spect
```

**Salidas:**

| Artefacto | Ruta |
|-----------|------|
| Checkpoint AAE SPECT | `result/exp_nm_s1_spect/AAE.pth.tar` |
| Latentes SPECT | `data/latent_Abeta/*.nii.gz` |
| Curvas etapa 1 | `result/exp_nm_s1_spect/stage1_abeta_autoencoder_*` |

Comprobar:

```bash
ls data/latent_Abeta | wc -l   # debe ser 148 (train codificado en E1)
```

### Análisis de embeddings SPECT (mismo split train/test que E2)

Clasificación y PCA/t-SNE sobre `data/latent_Abeta/`, usando los **mismos 118 train + 30 test** que
`result/exp_nm_s1_mri/phase1_mri/embedding_analysis/embedding_eval_split.csv`:

```bash
python3 code/analyze_spect_embeddings.py --exp exp_nm_s1_spect
```

Salida: `result/exp_nm_s1_spect/embedding_analysis/` (paralelo a MRI).

Comparar tablas SPECT vs MRI:

```bash
python3 scripts/compare_spect_mri_embedding_analysis.py
```

Nota: en latentes 76k-d, `SVM_RBF`/`SVM_linear` se omiten por defecto (muy lentos); KNN y
LogisticRegression usan el mismo protocolo que E2.

---

## 3. E2 — Fase 1 MRI (réplica + embeddings)

Pipeline dedicado (no usar `ILCLDM_RUN_STAGE1_MRI_PRETRAIN=1` en `main.py`; omite E1).

```bash
# Mismos ILCLDM_EPOCHS / LR que arriba; opcional overrides:
# export ILCLDM_EPOCHS=1000
scripts/run_experiment.sh s1_mri exp_nm_s1_mri
```

**Salidas:**

| Artefacto | Ruta |
|-----------|------|
| Checkpoint MRI AAE | `result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar` |
| Embeddings / análisis | `result/exp_nm_s1_mri/phase1_mri/` |

---

## 4. E3 — Fase 2 baseline (MRI crudo, sin encoder MRI preentrenado)

Condicionamiento: `DiffusionConditioner` sobre MRI cropeado (`ILCLDM_MRI_ENCODER_MODE=raw`).

**Requisito:** E1 terminado (`data/latent_Abeta/` poblado y `result/exp_nm_s1_spect/AAE.pth.tar` existe).

El script de stage 2 carga el AAE SPECT congelado. Por defecto busca `result/<ILCLDM_EXP>/AAE.pth.tar`; para usar el de **E1** sin copiar:

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
scripts/run_experiment.sh s2_baseline exp_nm_s2_baseline
```

Alternativa: copiar el checkpoint a `result/exp_nm_s2_baseline/AAE.pth.tar` (comportamiento legacy).

**Salidas:** `result/exp_nm_s2_baseline/Unet.pth.tar`, `loss_curve.csv`, `validation.csv`, `test.csv`.

---

## 5. E5 — Fase 2 con encoder MRI **congelado** (solo entrena UNet)

```bash
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar

mkdir -p result/exp_nm_s2_mri_frozen
cp result/exp_nm_s1_spect/AAE.pth.tar result/exp_nm_s2_mri_frozen/AAE.pth.tar

scripts/run_experiment.sh s2_mri_frozen exp_nm_s2_mri_frozen
```

Modo interno: `ILCLDM_MRI_ENCODER_MODE=frozen`.

---

## 5b. E5 v2 — Fase 2 frozen v2 (latent loss, high-t, ROI, early stop latente)

Mejoras sobre E5: pérdida directa en latente, sobremuestreo de timesteps altos (`t>700`), aux image en cada step con peso ROI (basal ganglia), y checkpoint por MSE latente en validación. DDIM sigue en **100 steps**.

**ROI:** `data/mask_basal_ganglia.nii.gz` (MNI, mismo `crop()` que MRI/SPECT; no requiere realineación).

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
scripts/run_experiment.sh s2_mri_frozen_v2 exp_nm_s2_mri_frozen_v2
```

Variables de entorno (preset en `s2_mri_frozen_v2`; override opcional):

| Variable | Default v2 |
|----------|------------|
| `ILCLDM_LATENT_LOSS` | `1` |
| `ILCLDM_LATENT_LOSS_LAMBDA` | `0.5` |
| `ILCLDM_AUX_IMAGE_EVERY_N` | `1` |
| `ILCLDM_AUX_IMAGE_LAMBDA` | `0.5` |
| `ILCLDM_AUX_IMAGE_ALWAYS_HIGH_T` | `1` |
| `ILCLDM_HIGH_T_SAMPLE_PROB` | `0.5` |
| `ILCLDM_HIGH_T_THRESHOLD` | `700` |
| `ILCLDM_ROI_MASK_PATH` | `data/mask_basal_ganglia.nii.gz` |
| `ILCLDM_ROI_WEIGHT_ALPHA` | `3.0` |
| `ILCLDM_USE_ROI_LOSS` | `1` |
| `ILCLDM_EARLY_STOP_LATENT_METRIC` | `1` |

**Salidas adicionales:** `validation_latent.csv`, `test_latent.csv`, columnas `latent_loss` en `loss_curve.csv`, `PSNR_roi`/`SSIM_roi` en `validation.csv`/`test.csv`.

**Inferencia (100 steps, default):**

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
python inference.py --exp exp_nm_s2_mri_frozen_v2 --run-id nm_stage2_v2 \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

`reconstruction_metrics.csv` incluye `PSNR_roi` y `SSIM_roi` cuando la máscara está disponible.

---

## 5c. Ablaciones frozen (registro + trazabilidad)

Para aislar el efecto de cada cambio de `frozen_v2`, las ablaciones parten de `frozen_v1` (encoder MRI congelado, early stop por imagen) y **cambian una sola cosa**. La definición vive en [`experiments/registry.json`](experiments/registry.json) (fuente única de verdad); el código base no se edita, solo se inyectan variables `ILCLDM_*`.

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar

# Una ablación = una clave del registro. --gpu fija CUDA_VISIBLE_DEVICES.
python3 scripts/run_ablation.py abl_latloss01 --gpu 1
# o vía run_experiment.sh:
scripts/run_experiment.sh abl abl_latloss01
```

Ablaciones definidas (cambio único sobre `base_frozen`):

| Clave | Carpeta `result/` | Cambio único |
|-------|-------------------|--------------|
| `abl_latloss01` | `exp_nm_s2_frozen_abl_latloss01` | `ILCLDM_LATENT_LOSS=1`, `ILCLDM_LATENT_LOSS_LAMBDA=0.1` |
| `abl_hight05` | `exp_nm_s2_frozen_abl_hight05` | `ILCLDM_HIGH_T_SAMPLE_PROB=0.5`, `ILCLDM_HIGH_T_THRESHOLD=700` |
| `abl_auxfreq5` | `exp_nm_s2_frozen_abl_auxfreq5` | `ILCLDM_AUX_IMAGE_EVERY_N=5`, `ILCLDM_AUX_IMAGE_LAMBDA=0.25` |

**Ejecución en paralelo** (cada proceso tiene su entorno y su carpeta `result/<exp>/`, sin interferencia):

```bash
python3 scripts/run_ablation.py abl_latloss01 --gpu 1
python3 scripts/run_ablation.py abl_hight05   --gpu 3
python3 scripts/run_ablation.py abl_auxfreq5  --gpu 5
```

**Trazabilidad:** cada corrida escribe `result/<exp>/run_manifest.json` (flags activos, seed, commit git, GPU). El notebook [`code/stage2_experiments_report.ipynb`](code/stage2_experiments_report.ipynb) deriva la comparación del registro y solo incluye experimentos con resultados en disco (`available_comparison()`), mostrando una tabla de configuración (`manifest_summary_table()`).

**Inferencia** (igual que frozen, usando el `run_id` del registro; 100 steps por defecto):

```bash
python inference.py --exp exp_nm_s2_frozen_abl_latloss01 --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --run-id abl_latloss01 --save-synthetic-spect --splits train test
```

> **Limitación de checkpoints:** se guarda solo el "best" según el criterio de early stopping (un único `Unet.pth.tar`). No se conservan pesos de épocas intermedias.

---

## 6. E4 — Fase 2 con encoder MRI **fine-tune** (UNet + encoder)

```bash
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar
export ILCLDM_MRI_ENCODER_FINETUNE_LR_SCALE=0.25

mkdir -p result/exp_nm_s2_mri_finetune
cp result/exp_nm_s1_spect/AAE.pth.tar result/exp_nm_s2_mri_finetune/AAE.pth.tar

scripts/run_experiment.sh s2_mri_finetune exp_nm_s2_mri_finetune
```

Modo interno: `ILCLDM_MRI_ENCODER_MODE=finetune` (LR del encoder = `ILCLDM_LR * FINETUNE_LR_SCALE`).

---

## 7. Inferencia y métricas en test

Cada experimento stage-2 debe usar el modo MRI coherente con el entrenamiento (`--mri-encoder-mode`):

| Experimento | `--mri-encoder-mode` | Checkpoint MRI |
|-------------|----------------------|----------------|
| `exp_nm_s2_baseline` | `raw` | (ninguno; `DiffusionConditioner` en UNet) |
| `exp_nm_s2_mri_frozen` | `frozen` | `result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar` |
| `exp_nm_s2_mri_frozen_v2` | `frozen` | `result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar` |
| `exp_nm_s2_mri_finetune` | `finetune` | `result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar` |

Si E4 terminó antes de guardar el encoder afinado:

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
python3 scripts/recover_finetune_mri_encoder.py --exp exp_nm_s2_mri_finetune --epochs 50
```

Inferencia por experimento (reconstrucciones, latentes `sampled_latent_final`, ML train→test):

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar

python inference.py --exp exp_nm_s2_baseline --mri-encoder-mode raw \
  --run-id nm_stage2_v1 --save-synthetic-spect --splits train test

python inference.py --exp exp_nm_s2_mri_frozen --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --run-id nm_stage2_v1 --save-synthetic-spect --splits train test

python inference.py --exp exp_nm_s2_mri_frozen_v2 --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --run-id nm_stage2_v2 --save-synthetic-spect --splits train test

python inference.py --exp exp_nm_s2_mri_finetune --mri-encoder-mode finetune \
  --mri-encoder-checkpoint result/exp_nm_s2_mri_finetune/MRI_Encoder_finetuned.pth.tar \
  --run-id nm_stage2_v1 --save-synthetic-spect --splits train test
```

Batch en GPU 6 (recomendado si está libre; reanuda `.npz` existentes):

```bash
CUDA_VISIBLE_DEVICES=6 RUN_ID=nm_stage2_v1 bash scripts/run_stage2_gpu6_pipeline.sh
```

Alternativa por experimento:

```bash
CUDA_VISIBLE_DEVICES=6 RUN_ID=nm_stage2_v1 bash scripts/run_stage2_inference.sh
```

Salidas: `result/<exp>/inference/<run_id>/` y `result/stage2_inference_summary/`.

---

## Orden recomendado (resumen)

```text
preflight → E1 (SPECT) → E2 (MRI) → E3 (baseline LDM)
                              ↘
                    E5 (frozen)  y/o  E4 (finetune)  y/o  E5 v2 (frozen_v2)
                              ↘
                         inference por --exp
```

E5 y E4 pueden lanzarse en paralelo en GPUs distintas si copias el AAE de E1 en cada `result/<exp>/`.

---

## Visualización rápida (z-score)

```bash
python code/visualize_pair_example.py
# o en notebook:
# from viz_pairs import plot_subject
# plot_subject(Path("data"), "3102")
```

Usar escala simétrica para MRI; no filtrar solo `voxels > 0`.

---

## Troubleshooting

| Problema | Acción |
|----------|--------|
| `FileNotFoundError` en un ID | `python3 scripts/sync_data_info_to_data.py` |
| `latent_Abeta` vacío en etapa 2 | Completar E1 primero |
| `MRI_ENCODER_CHECKPOINT` no encontrado | Completar E2; revisar ruta en E5/E4 |
| AAE SPECT no encontrado en E3/E5/E4 | `cp result/exp_nm_s1_spect/AAE.pth.tar result/<exp_s2_*>/` |
| Pérdidas raras tras cambiar datos | Vaciar `data/latent_*` y repetir `encoding()` (E1) |

---

## Referencia de modos (`scripts/run_experiment.sh`)

| Comando | Descripción |
|---------|-------------|
| `preflight` | `preflight_data.py` + `smoke_test_fsierra.py` |
| `s1_spect [exp]` | Etapa 1 SPECT + encode |
| `s1_mri [exp]` | Etapa 1 MRI (`run_phase1_mri_experiment.py`) |
| `s2_baseline [exp]` | LDM, MRI crudo |
| `s2_mri_frozen [exp]` | LDM, encoder MRI congelado |
| `s2_mri_frozen_v2 [exp]` | LDM frozen v2 (latent loss, high-t, ROI, early stop latente) |
| `s2_mri_finetune [exp]` | LDM, encoder MRI entrenable |
| `abl <registry_key>` | Ablación frozen desde `experiments/registry.json` |

Documentación ampliada: [DEVCONTAINER.md](DEVCONTAINER.md).
