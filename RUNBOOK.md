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

## 5d. Wave h05 (warm-start desde `abl_hight05`)

Cuatro experimentos derivados del baseline `abl_hight05`, cada uno con **un solo delta**, warm-start desde `result/exp_nm_s2_frozen_abl_hight05/Unet.pth.tar`. Definidos en [`experiments/registry.json`](experiments/registry.json) con trazabilidad (`parent`, `init_checkpoint`, `run_manifest.json`).

| Clave | Carpeta `result/` | Delta único |
|-------|-------------------|-------------|
| `h05_ht08` | `exp_nm_s2_frozen_h05_ht08` | `ILCLDM_HIGH_T_SAMPLE_PROB=0.8` |
| `h05_roi3ch` | `exp_nm_s2_frozen_h05_roi3ch` | `ILCLDM_UNET_ROI_CHANNEL=1` (3.er canal UNet) |
| `h05_nolabel` | `exp_nm_s2_frozen_h05_nolabel` | `ILCLDM_USE_TRAIN_LABEL=0` |
| `h05_aux5` | `exp_nm_s2_frozen_h05_aux5` | `ILCLDM_AUX_IMAGE_EVERY_N=5`, `ILCLDM_AUX_IMAGE_LAMBDA=0.25` |

**Requisito:** el checkpoint de `abl_hight05` debe existir antes de lanzar la wave.

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar

python3 scripts/run_ablation.py h05_ht08    --gpu 0
python3 scripts/run_ablation.py h05_roi3ch  --gpu 1
python3 scripts/run_ablation.py h05_nolabel --gpu 2
python3 scripts/run_ablation.py h05_aux5    --gpu 3
```

**Inferencia** (el UNet de 3 canales se infiere de `run_manifest.json`; no hace falta exportar `ILCLDM_UNET_ROI_CHANNEL`):

```bash
python inference.py --exp exp_nm_s2_frozen_h05_roi3ch --run-id h05_roi3ch \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

---

## 5f. Baseline calibrado (`calib_roi3ch_uni`)

**Primera E2 coherente con calibración latente** `(z−μ)/σ` antes de difundir y desnormalización antes del decoder AAE congelado de E1. Contrasta con `h05_roi3ch` (latentes crudos en difusión + muestreo high-t 0.5 + warm-start).

| Aspecto | `h05_roi3ch` (anterior) | `calib_roi3ch_uni` (nuevo) |
|---------|-------------------------|----------------------------|
| Calibración latente | no (UNet en espacio crudo) | **sí** (`ILCLDM_LATENT_CALIBRATION=1`) |
| Muestreo `t` | `HIGH_T_SAMPLE_PROB=0.5` | **uniforme** (`HIGH_T_SAMPLE_PROB=0`) |
| ROI 3.er canal | sí | sí |
| E1 SPECT/MRI | congelados | congelados (mismos checkpoints) |
| Latentes train | `data/latent_Abeta/` crudos | **mismos archivos** (μ/σ en `data_info/latent_stats_latent_Abeta.json`) |
| UNet init | warm-start `abl_hight05` | **scratch** (sin `init_checkpoint`) |

**Clave registry:** `calib_roi3ch_uni` → `result/exp_nm_s2_frozen_calib_roi3ch_uni/`

**No requiere repetir E1:** los `.nii.gz` en `data/latent_Abeta/` son salidas crudas del encoder; la normalización ocurre al cargar en E2.

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
python3 scripts/run_ablation.py calib_roi3ch_uni --gpu 0
```

**Inferencia:**

```bash
python3 inference.py --exp exp_nm_s2_frozen_calib_roi3ch_uni --run-id calib_roi3ch_uni \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

Re-ejecutar `code/auditoria_diffusion.ipynb` con `AUDIT_EXP=exp_nm_s2_frozen_calib_roi3ch_uni` tras entrenar para comparar one-step / DDIM / calibración.

---

## 5g. Calibrado + high-t (`h05_calib_roi3ch`)

**Celda factorial que faltaba:** misma calibración latente que `calib_roi3ch_uni`, pero con **warm-start** desde su UNet y **sobremuestreo high-t 50%** en `t > 700` (receta `h05_roi3ch`).

| Aspecto | `calib_roi3ch_uni` | `h05_calib_roi3ch` |
|---------|-------------------|---------------------|
| Calibración latente | sí | sí |
| Muestreo `t` | uniforme (`HIGH_T_SAMPLE_PROB=0`) | **0.5 en t>700** |
| UNet init | scratch | **warm-start `calib_roi3ch_uni`** |
| ROI 3.er canal | sí | sí |

**Prerrequisito:** `result/exp_nm_s2_frozen_calib_roi3ch_uni/Unet.pth.tar` debe existir.

**Clave registry:** `h05_calib_roi3ch` → `result/exp_nm_s2_frozen_h05_calib_roi3ch/`

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar
python3 scripts/run_ablation.py h05_calib_roi3ch --gpu <N>
```

**Inferencia** (el manifest guardará `LATENT_CALIBRATION=1`; `inference.py` lo aplica desde `run_manifest.json`):

```bash
python3 inference.py --exp exp_nm_s2_frozen_h05_calib_roi3ch --run-id h05_calib_roi3ch \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

Tras entrenar e inferir, `h05_calib_roi3ch` está en `WAVE_ORDER` de `code/stage2_experiments_report.ipynb` — re-ejecutar el notebook (§6 respeta `LATENT_CALIBRATION` por experimento).

---

## 5e. Wave roi3ch (split NHY + 3 derivados)

Segunda wave centrada en **`h05_roi3ch`** (3.er canal UNet, clasificación PD). Cinco claves en [`experiments/registry.json`](experiments/registry.json): baseline `h05_roi3ch` + cuatro nuevos experimentos.

| Clave | Carpeta `result/` | Split | Delta principal |
|-------|-------------------|-------|-----------------|
| `h05_roi3ch` | `exp_nm_s2_frozen_h05_roi3ch` | primario | baseline wave h05 (ROI 3ch) |
| `abl_hight05_nhy` | `exp_nm_s2_frozen_abl_hight05_nhy` | **nhy_v1** | high-t 0.5, scratch (análogo `abl_hight05`) |
| `h05_roi3ch_nhy` | `exp_nm_s2_frozen_h05_roi3ch_nhy` | **nhy_v1** | warm-start `abl_hight05_nhy` + ROI 3ch (análogo `h05_roi3ch`) |
| `calib_roi3ch_nhy` | `exp_nm_s2_frozen_calib_roi3ch_nhy` | **nhy_v1** | calibración latente + ROI 3ch + t uniforme, scratch (análogo `calib_roi3ch_uni`) |
| `roi3ch_nhy` | `exp_nm_s2_frozen_roi3ch_nhy` | **nhy_v1** | pipeline E1→E2→E3 completo, **sin** warm-start (baseline cold-start) |
| `roi3ch_nolabel` | `exp_nm_s2_frozen_roi3ch_nolabel` | primario | `USE_TRAIN_LABEL=0`, warm-start `h05_roi3ch` |
| `roi3ch_ht08` | `exp_nm_s2_frozen_roi3ch_ht08` | primario | `HIGH_T_SAMPLE_PROB=0.8`, warm-start `h05_roi3ch` |
| `roi3ch_cls` | `exp_nm_s2_frozen_roi3ch_cls` | primario | cls loss en `x0_hat`, early stop **solo** `balanced_accuracy` |

**Métrica primaria del reporte:** balanced accuracy (clasificación PD/control). PSNR/SSIM quedan como diagnóstico.

### Pista A — split NHY (artefactos aislados)

**No modifica** `data_info/train.txt`, `test.txt` ni `data/latent_Abeta/` del split primario.

Hay dos variantes de E3 en NHY:

| Ruta | Claves | Uso |
|------|--------|-----|
| **Cadena equivalente a primario** | `abl_hight05_nhy` → `h05_roi3ch_nhy` | Comparación justa con `h05_roi3ch` (warm-start sin fuga) |
| **Cold-start** | `roi3ch_nhy` | Baseline ya corrido (ROI 3ch from scratch) |

**Paso 0 — Generar split** (sin GPU):

```bash
python3 scripts/generate_nhy_split.py --out data_info/splits/nhy_v1
# Revisar data_info/splits/nhy_v1/split_manifest.json y nhy_distribution.csv
```

**Variables de entorno** para toda la pista NHY (`source scripts/nhy_v1_env.sh`):

```bash
export ILCLDM_TRAIN_FILE=data_info/splits/nhy_v1/train.txt
export ILCLDM_TEST_FILE=data_info/splits/nhy_v1/test.txt
export ILCLDM_VALIDATION_FILE=data_info/splits/nhy_v1/validation.txt
export ILCLDM_LATENT_ABETA=data/latent_Abeta_nhy_v1/
```

**Paso 1 — E1 SPECT** (AAE + latentes del train NHY):

```bash
source scripts/nhy_v1_env.sh
export ILCLDM_EXP=exp_nm_s1_spect_nhy_v1/
export ILCLDM_RUN_STAGE1_MRI_PRETRAIN=0
export ILCLDM_RUN_STAGE2_DIFFUSION=0
scripts/run_experiment.sh s1_spect exp_nm_s1_spect_nhy_v1
# → result/exp_nm_s1_spect_nhy_v1/AAE.pth.tar
# → data/latent_Abeta_nhy_v1/*.nii.gz (~148 IDs train nhy_v1)
```

**Paso 2 — E1 MRI**:

```bash
source scripts/nhy_v1_env.sh
scripts/run_experiment.sh s1_mri exp_nm_s1_mri_nhy_v1
# → result/exp_nm_s1_mri_nhy_v1/phase1_mri/checkpoints/MRI_AAE.pth.tar
```

**Paso 3a — E3 cadena NHY** (recomendado, análogo `abl_hight05` → `h05_roi3ch`):

```bash
# 3a.1 high-t scratch (2ch UNet)
python3 scripts/run_ablation.py abl_hight05_nhy --gpu <N>
# → result/exp_nm_s2_frozen_abl_hight05_nhy/Unet.pth.tar

# 3a.2 ROI 3ch warm-start desde abl_hight05_nhy
python3 scripts/run_ablation.py h05_roi3ch_nhy --gpu <N>
# → result/exp_nm_s2_frozen_h05_roi3ch_nhy/Unet.pth.tar
```

**Paso 3b — E3 `calib_roi3ch_nhy`** (calibrado + ROI 3ch, scratch; análogo `calib_roi3ch_uni` en split NHY):

```bash
python3 scripts/run_ablation.py calib_roi3ch_nhy --gpu <N>
# → result/exp_nm_s2_frozen_calib_roi3ch_nhy/Unet.pth.tar
# Stats calibración: data_info/latent_stats_latent_Abeta_nhy_v1.json (auto al entrenar)
```

**Paso 3c — E3 `roi3ch_nhy`** (cold-start, opcional / ya corrido):

```bash
python3 scripts/run_ablation.py roi3ch_nhy --gpu <N>
```

**Paso 4 — Inferencia** (`h05_roi3ch_nhy` o `roi3ch_nhy`):

```bash
source scripts/nhy_v1_env.sh
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect_nhy_v1/AAE.pth.tar
export ILCLDM_MRI_ENCODER_CHECKPOINT=result/exp_nm_s1_mri_nhy_v1/phase1_mri/checkpoints/MRI_AAE.pth.tar

# Cadena NHY (equivalente h05_roi3ch)
python3 inference.py --exp exp_nm_s2_frozen_h05_roi3ch_nhy --run-id h05_roi3ch_nhy \
  --mri-encoder-mode frozen --splits train test --save-synthetic-spect

# Cold-start (baseline)
python3 inference.py --exp exp_nm_s2_frozen_roi3ch_nhy --run-id roi3ch_nhy \
  --mri-encoder-mode frozen --splits train test --save-synthetic-spect

# Calibrado NHY (análogo calib_roi3ch_uni)
python3 inference.py --exp exp_nm_s2_frozen_calib_roi3ch_nhy --run-id calib_roi3ch_nhy \
  --mri-encoder-mode frozen --splits train test --save-synthetic-spect
```

### Pista B — derivados en split primario (warm-start `h05_roi3ch`)

Requisito: checkpoint `result/exp_nm_s2_frozen_h05_roi3ch/Unet.pth.tar` debe existir.

```bash
export ILCLDM_AAE_CHECKPOINT=result/exp_nm_s1_spect/AAE.pth.tar

python3 scripts/run_ablation.py roi3ch_nolabel --gpu 0
python3 scripts/run_ablation.py roi3ch_ht08    --gpu 1
python3 scripts/run_ablation.py roi3ch_cls     --gpu 2
```

`roi3ch_cls` guarda `ClsHead.pth.tar` junto al UNet y usa early stopping por `validation_cls.csv` (balanced accuracy).

**Inferencia** (split primario por defecto):

```bash
python3 inference.py --exp exp_nm_s2_frozen_roi3ch_nolabel --run-id roi3ch_nolabel \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

Reporte notebook: sección **Wave roi3ch** en [`code/stage2_experiments_report.ipynb`](code/stage2_experiments_report.ipynb); export CSV en `result/stage2_roi3ch_wave_summary/`.

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

## Auditoría rápida de difusión (notebook)

Diagnóstico visual **sin reentrenar**: schedule coseno (`s`), one-step vs DDIM, calibración de latentes, ε-error en t altos.

```bash
cd /path/to/IL-CLDM
# Jupyter / VS Code: abrir code/auditoria_diffusion.ipynb
# Perfil rápido por defecto: 2 sujetos test, exp exp_nm_s2_frozen_h05_roi3ch
jupyter nbconvert --to notebook --execute code/auditoria_diffusion.ipynb \
  --output auditoria_diffusion_executed.ipynb
```

**Knobs** (celda de configuración del notebook): `AUDIT_EXP`, `MAX_SUBJECTS`, `FIXED_T_VALUES`, `COSINE_S_VALUES`, `DDIM_STEPS_LIST`.

**Salidas:** `result/auditoria_diffusion/*.png` y `*.csv`.

Requisitos: checkpoints E1 (`AAE`) y E2 (`Unet.pth.tar`) del experimento; latentes en `data/latent_Abeta/` para stats de calibración (`data_info/latent_stats_latent_Abeta.json`).

---

## Experimento ROI (`data_rois`)

Patches ya recortados (bbox ROI + margen). **No** se aplica el crop IL-CLDM `[10:170,…]` en carga (`ILCLDM_SKIP_CROP=1`). Shape en disco tras reproc: `(96, 112, 114)`; padding AAE/UNet a `(96, 112, 116)`; grid latente `(24, 28, 29)`. (Histórico pre-reproc: `(105, 102, 102)` → pad `(108, 104, 104)`.)

### Preflight ROI

```bash
python3 scripts/preflight_data_rois.py
python3 scripts/aggregate_roi_unet_mask.py   # máscara debe coincidir con 96,112,114
python3 scripts/smoke_test_rois.py
```

### Entorno ROI (E1 / E2)

```bash
source scripts/roi_profile_env.sh
# o export manual: ILCLDM_SKIP_CROP=1, whole_* bajo data_rois/, etc.
```

| Variable | Valor ROI |
|----------|-----------|
| `ILCLDM_SKIP_CROP` | `1` |
| `ILCLDM_WHOLE_MRI` / `ABETA` | `data_rois/whole_*` |
| `ILCLDM_LATENT_ABETA` | `data_rois/latent_Abeta/` |
| `ILCLDM_LATENT_STATS_PATH` | `data_info/latent_stats_latent_Abeta_rois.json` |
| `ILCLDM_ROI_MASK_PATH` | `data_rois/roi_mask_unet.nii.gz` |
| `ILCLDM_ROI_VOLUME_SHAPE` | `96,112,114` |

### E1 ROI (desde cero, no reutilizar full-brain)

```bash
export CUDA_VISIBLE_DEVICES=6   # ajustar GPU
bash scripts/run_e1_rois.sh
```

Equivalente paso a paso:

```bash
source scripts/roi_profile_env.sh
export ILCLDM_EXP=exp_nm_s1_spect_rois/
scripts/run_experiment.sh s1_spect exp_nm_s1_spect_rois
python3 code/run_phase1_mri_experiment.py --exp exp_nm_s1_mri_rois
python3 -c "from dataset import get_latent_stats; print(get_latent_stats())"
```

Checkpoints esperados:

- `result/exp_nm_s1_spect_rois/AAE.pth.tar`
- `result/exp_nm_s1_mri_rois/phase1_mri/checkpoints/MRI_AAE.pth.tar`
- `data_rois/latent_Abeta/*.nii.gz` (train encode)
- `data_info/latent_stats_latent_Abeta_rois.json`

### E2 ROI — `calib_roi3ch_uni_rois`

Receta: calibración latente `(z−μ)/σ`, desnormalizar antes del decoder; `t` uniforme; UNet scratch; 3.er canal ROI; label en train.

```bash
source scripts/roi_profile_env.sh
python3 scripts/run_ablation.py calib_roi3ch_uni_rois --gpu 6
```

Registry: `experiments/registry.json` → `calib_roi3ch_uni_rois` (base `base_calib_roi3ch_uni_rois`).

Resultados: `result/exp_nm_s2_frozen_calib_roi3ch_uni_rois/`.

### E2 ROI + conditioner `xmodal_h64` (`xm64_z18` / `xm64_h64` / `xm64_h64_gap1`)

Tres variantes aisladas del baseline `calib_roi3ch_uni_rois`. MRI encoder = checkpoint E1 `exp_nm_s1_mri_rois_xmodal_h64`. **No** modifican el run `calib_roi3ch_uni_rois`.

| Clave registry | Espacio difusión | Notas |
|----------------|------------------|-------|
| `calib_roi3ch_uni_rois_xm64_z18` | bottleneck z18k (1ch) | Solo cambia MRI ckpt; mismos latentes `data_rois/latent_Abeta/` |
| `calib_roi3ch_uni_rois_xm64_h64` | h64 (64ch) | Cache `data_rois/latent_Abeta_h64/`; UNet in=129; aux decode off |
| `calib_roi3ch_uni_rois_xm64_h64_gap1` | channel-mean GAP → 1ch | Mismo cache h64; stats gap1; UNet como z18k |

**Prerequisito h64** (ya ejecutado / re-ejecutable en GPU 7):

```bash
source scripts/roi_profile_env.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/encode_spect_h64_rois.sh
# → data_rois/latent_Abeta_h64/*.npy
# → data_info/latent_stats_latent_Abeta_rois_h64.json
# → data_info/latent_stats_latent_Abeta_rois_h64_gap1.json
```

**Lanzar E2** (cuando se autorice; B=2 en registry):

```bash
source scripts/roi_profile_env.sh
python3 scripts/run_ablation.py calib_roi3ch_uni_rois_xm64_z18 --gpu <N>
python3 scripts/run_ablation.py calib_roi3ch_uni_rois_xm64_h64 --gpu <N>
python3 scripts/run_ablation.py calib_roi3ch_uni_rois_xm64_h64_gap1 --gpu <N>
```

Flag: `ILCLDM_DIFFUSION_FEATURE_SPACE={z18k,h64,h64_gap1}`. Decode SPECT desde h64 (bonus): `ILCLDM_DIFFUSION_H64_DECODE_BONUS=1` (h64 → Conv64→1 frozen → decoder). Eval primaria: embeddings discriminativos, no PSNR.

### E1 MRI contrastivo espacial (`data_rois`)

Variantes de **E1 MRI** aisladas del baseline `exp_nm_s1_mri_rois`. Contrastiva **directa** sobre el mapa espacial (`L2(flatten)`), sin GAP ni proyector MLP. Prefijo: `exp_nm_s1_mri_rois_`. No reutilizar nombres cortos viejos (`…_supcon` / `…_xmodal` sin sufijo de features).

| Clave corta | `--exp` | Modo | Features contrastivas |
|-------------|---------|------|------------------------|
| `supcon_z18k` | `exp_nm_s1_mri_rois_supcon_z18k` | SupCon | flatten bottleneck `(1,27,26,26)` → 18 252-D |
| `supcon_h64` | `exp_nm_s1_mri_rois_supcon_h64` | SupCon | flatten pre-bottleneck `(64,27,26,26)` |
| `xmodal_z18k` | `exp_nm_s1_mri_rois_xmodal_z18k` | InfoNCE | MRI+SPECT flatten bottleneck 18 252-D |
| `xmodal_h64` | `exp_nm_s1_mri_rois_xmodal_h64` | InfoNCE | MRI+SPECT flatten mapa 64-ch |

Baseline recon: `exp_nm_s1_mri_rois`.

**Prerrequisitos:**

- `source scripts/roi_profile_env.sh`
- E1 SPECT ROI: `result/exp_nm_s1_spect_rois/AAE.pth.tar` (obligatorio para xmodal)
- Warm-start: `result/exp_nm_s1_mri_rois/phase1_mri/checkpoints/MRI_AAE.pth.tar`

**SupCon** (etiqueta clínica; batches 4+4; positivos = misma clase):

```bash
source scripts/roi_profile_env.sh
CUDA_VISIBLE_DEVICES=<N> bash scripts/run_e1_mri_rois_supcon_z18k.sh
# o: bash scripts/run_e1_mri_rois_supcon_h64.sh
```

**Cross-modal** (pares mismo ID; SPECT frozen; sin balance por label; 1 positivo + 7 negativos con B=8):

```bash
source scripts/roi_profile_env.sh
CUDA_VISIBLE_DEVICES=<N> bash scripts/run_e1_mri_rois_xmodal_z18k.sh
# o: bash scripts/run_e1_mri_rois_xmodal_h64.sh
```

**Defaults:** `batch_size=8`, `λ=20`, `τ=0.5` (`--contrastive-temp`), `--contrastive-features {z18k,h64}`, `projector=false`. Checkpoint por **mínimo `val_contrastive_loss`** (`selection_metric=neg_val_contrastive_loss`). Eval/clasificación sobre el mismo flatten que la loss. Override: `BATCH_SIZE=`, `SUPCON_LAMBDA=` / `XMODAL_LAMBDA=`, `CONTRASTIVE_TEMP=`.

**Salidas:** `result/<exp>/phase1_mri/` (`run_config.json` con `contrastive_features`, `contrastive_temp`, `projector`); embeddings `z18k` como NIfTI, `h64` como `.npy`; retrieval xmodal en `…/classification/crossmodal_retrieval.json`.

**Criterios:** `val_contrastive_loss` ≪ chance (`ln(B)` / `ln(B−1)`); SupCon within-cos < between-cos o bal.acc > baseline; xmodal top-1 retrieval > azar. Prioridad analítica: **z18k**.

### E1 ROI SPECT 4 mm + MRI xmodal h64 (`data_rois/whole_Abeta_4mm`)

Pipeline **aislado** para SPECT ROI **4 mm** (no `data_rois/whole_Abeta` 18 mm, no `data/whole_Abeta_4mm` full-brain). No pisa `spect_rois_reproc`, `latent_Abeta_reproc/` ni E2. No reanudar `exp_nm_s1_mri_rois_reproc_xmodal_h64` (18 mm, abortado).

| Paso | `--exp` / script | Datos SPECT | Notas |
|------|------------------|-------------|--------|
| A — E1 SPECT | `exp_nm_s1_spect_rois_4mm` | `data_rois/whole_Abeta_4mm` | Latentes `data_rois/latent_Abeta_4mm/`; stats `data_info/latent_stats_latent_Abeta_rois_4mm.json` |
| B — xmodal h64 | `exp_nm_s1_mri_rois_4mm_xmodal_h64` | mismo 4 mm | Init MRI `exp_nm_s1_mri_rois_reproc`; SPECT frozen = AAE del paso A |

MRI: `data_rois/whole_MRI` (96×112×114; pad 96×112×116; h64 `64×24×28×29`). GPU por defecto **5**. `roi_profile_env.sh` no se modifica: los scripts exportan `ILCLDM_WHOLE_ABETA` **después** del `source`.

```bash
source scripts/roi_profile_env.sh
# cadena (SPECT luego xmodal) en GPU 5:
CUDA_VISIBLE_DEVICES=5 bash scripts/run_e1_rois_4mm_spect_then_xmodal.sh
# o por pasos:
CUDA_VISIBLE_DEVICES=5 bash scripts/run_e1_spect_rois_4mm.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/run_e1_mri_rois_4mm_xmodal_h64.sh
```

Misma receta xmodal 105³ (`λ=20`, `τ=0.5`, `--contrastive-features h64`). **Gate:** `run_config` con shape 96³ y SPECT ckpt `spect_rois_4mm`; BA flatten h64 claramente > recon reproc (~0.64). Referencia 105³ ~0.92.

### Clasificador MobileNetV2-3D sobre recon AAE SPECT 4 mm

Independiente del xmodal (puede correr en paralelo). Entrena un **MobileNetV2 inflado a 3D** sobre `recon = decoder(encoder(spect))` del AAE congelado — un CE por volumen (no voto de cortes). No toca E2 ni `registry.json`.

| | |
|--|--|
| SPECT | `data_rois/whole_Abeta_4mm` |
| AAE frozen | `result/exp_nm_s1_spect_rois_4mm/AAE.pth.tar` |
| Salida | `result/exp_nm_cls_spect_aae_mnet3d/` (`best.pt`, `config.json`, `metrics_valid.json`, `metrics_test.json`, `recon_cache/`) |
| GPU | **7** |

```bash
# GPU 7 is the launcher default (override with CLS_GPU=N if needed)
bash scripts/run_cls_spect_aae_mnet3d_4mm.sh
# o nohup:
mkdir -p logs
setsid nohup bash scripts/run_cls_spect_aae_mnet3d_4mm.sh \
  > logs/cls_spect_aae_mnet3d_4mm.log 2>&1 < /dev/null &
```

**Gate (antes de FST):** BA val/test claramente > 0.5 sobre reconstrucciones. Si no supera el azar, el decoder no conserva señal diagnóstica útil para ponderar FST con 0.8 CE.

**Run GPU 7 (completado):** early-stop época 24 (best época 9). Val/test **BA = 0.975**, ROC-AUC = 1.0 (`result/exp_nm_cls_spect_aae_mnet3d/metrics_*.json`). Gate azar **OK** (≫ 0.5; también > MRI recon ~0.64).

### Feature-space transfer (FST / ARAD) — repo hermano

El mapeo MRI h64 → SPECT h64 (UNet 64→64, sin difusión) vive en **`/workspace/IL-FST`**, no en este repo. Congela los gates de arriba (SPECT AAE 4 mm, MRI xmodal h64, MobileNet3D), loss \(0.2\,\mathrm{MSE}+0.8\,\mathrm{CE}\), GPU **6**. Ver `IL-FST/RUNBOOK_FST.md` y `result/fst_h64_arad/`. **No** mezclar con E2 ni con SPECT 18 mm / xmodal 105³.

```bash
cd /workspace/IL-FST && source env.sh
bash scripts/smoke_fst_frozen.sh
mkdir -p logs
setsid nohup bash scripts/run_fst_h64_arad.sh \
  > logs/fst_h64_arad.log 2>&1 < /dev/null &
```

---

## Visualización rápida (z-score)

```bash
python code/visualize_pair_example.py
# o en notebook:
# from viz_pairs import plot_subject
# plot_subject(Path("data"), "3102")
```

Usar escala simétrica para MRI; no filtrar solo `voxels > 0` (salvo modo `v2load` abajo).

---

## Reproc: datos nuevos + pares `_reproc` / `_reproc_v2load`

Réplicas del baseline calibrado `calib_roi3ch_uni` (full) y `calib_roi3ch_uni_rois` (ROI) sobre NIfTI reprocesados en `data/` y `data_rois/`. **No** sobrescriben los resultados legacy. Split original: `data_info/train.txt`, `validation.txt`, `test.txt`.

| Par | Carga (`ILCLDM_IMAGE_VOLUME_NORM`) | Claves E2 |
|-----|--------------------------------------|-----------|
| A `_reproc` | `legacy` — z-score MRI+SPECT (todo el volumen) | `calib_roi3ch_uni_reproc`, `calib_roi3ch_uni_rois_reproc` |
| B `_reproc_v2load` | `v2load` — MRI: μ/σ en vóxeles `>0`; SPECT: sin z-score | `calib_roi3ch_uni_reproc_v2load`, `calib_roi3ch_uni_rois_reproc_v2load` |

Cada par necesita **su propia** cadena E1 → latents → stats → E2 (no reutilizar AAE/latents entre `reproc` y `reproc_v2load`, ni con el baseline viejo).

### Cadena full — par A (`legacy`, SPECT `whole_Abeta_4mm`)

Full-brain con z-score MRI+SPECT (`legacy`). SPECT se lee de `data/whole_Abeta_4mm`.
El E1 MRI **no se reentrena**: se reutiliza `exp_nm_s1_mri` (mismos volúmenes `data/whole_MRI`).

```bash
cd /workspace/IL-CLDM
export ILCLDM_IMAGE_VOLUME_NORM=legacy
export ILCLDM_WHOLE_ABETA=data/whole_Abeta_4mm
export ILCLDM_WHOLE_MRI=data/whole_MRI
export ILCLDM_LATENT_ABETA=data/latent_Abeta_reproc/
export ILCLDM_LATENT_STATS_PATH=data_info/latent_stats_latent_Abeta_reproc.json

# 1) E1 SPECT + encode (desde whole_Abeta_4mm)
scripts/run_experiment.sh s1_spect exp_nm_s1_spect_reproc
# → result/exp_nm_s1_spect_reproc/AAE.pth.tar + data/latent_Abeta_reproc/

# 2) Stats de calibración (no hace falta s1_mri_reproc)
python3 -c "from dataset import get_latent_stats; print(get_latent_stats())"

# 3) E2 (registry: MRI checkpoint = exp_nm_s1_mri, SPECT path = whole_Abeta_4mm)
python3 scripts/run_ablation.py calib_roi3ch_uni_reproc --gpu <N>
# → result/exp_nm_s2_frozen_calib_roi3ch_uni_reproc/

# 4) Inferencia + clasificación
python3 inference.py \
  --exp exp_nm_s2_frozen_calib_roi3ch_uni_reproc \
  --run-id calib_roi3ch_uni_reproc \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
# → classification/classification_metrics.csv bajo result/.../inference/<run_id>/
```

### Cadena full — par B (`v2load`)

Igual que arriba cambiando sufijos y el flag de carga:

```bash
export ILCLDM_IMAGE_VOLUME_NORM=v2load
export ILCLDM_LATENT_ABETA=data/latent_Abeta_reproc_v2load/
export ILCLDM_LATENT_STATS_PATH=data_info/latent_stats_latent_Abeta_reproc_v2load.json

scripts/run_experiment.sh s1_spect exp_nm_s1_spect_reproc_v2load
scripts/run_experiment.sh s1_mri exp_nm_s1_mri_reproc_v2load
python3 -c "from dataset import get_latent_stats; print(get_latent_stats())"
python3 scripts/run_ablation.py calib_roi3ch_uni_reproc_v2load --gpu <N>
python3 inference.py \
  --exp exp_nm_s2_frozen_calib_roi3ch_uni_reproc_v2load \
  --run-id calib_roi3ch_uni_reproc_v2load \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri_reproc_v2load/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

### Cadena ROI — par A / B

```bash
source scripts/roi_profile_env.sh
# Par A:
export ILCLDM_IMAGE_VOLUME_NORM=legacy
export ILCLDM_LATENT_ABETA=data_rois/latent_Abeta_reproc/
export ILCLDM_LATENT_STATS_PATH=data_info/latent_stats_latent_Abeta_rois_reproc.json
export ILCLDM_ROI_VOLUME_SHAPE=96,112,114
export ILCLDM_EXP=exp_nm_s1_spect_rois_reproc/
python3 main.py   # E1 SPECT encode
python3 code/run_phase1_mri_experiment.py --exp exp_nm_s1_mri_rois_reproc
python3 -c "from dataset import get_latent_stats; print(get_latent_stats())"
python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc --gpu <N>
python3 inference.py \
  --exp exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc \
  --run-id calib_roi3ch_uni_rois_reproc \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri_rois_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test

# Par B: mismos comandos con sufijo _reproc_v2load, IMAGE_VOLUME_NORM=v2load,
# latent_Abeta_reproc_v2load, y clave registry calib_roi3ch_uni_rois_reproc_v2load.
```

`inference.py` restaura `IMAGE_VOLUME_NORM`, paths y split desde `run_manifest.json` (escrito en E2). Comparar: baseline legacy vs `_reproc` (efecto datos) y `_reproc` vs `_reproc_v2load` (efecto carga).

### Evaluación NHY3 sobre embeddings reproc (`calib_roi3ch_uni_rois_reproc_nhy3`)

Extensión de **análisis** (no reentrena LDM) sobre el mejor binario `calib_roi3ch_uni_rois_reproc`. Reusa `inference/.../reps/` + `metadata.csv`; remapa etiquetas desde `data_info/splits/nhy_v1/nhy_distribution.csv` (`control`→0, `pd_nhy1`→1, `pd_nhy2`→2). **Mantiene el split train/test de la inferencia** (no el `split` de `nhy_distribution.csv`). Salida en `classification_nhy3/` — **no** toca `classification/` binaria.

| analysis_id | Parent | Tipo |
|-------------|--------|------|
| `calib_roi3ch_uni_rois_reproc_nhy3` | `calib_roi3ch_uni_rois_reproc` | `embedding_eval` (registry → `embedding_evals`) |

```bash
python3 scripts/classify_embeddings_nhy3.py
python3 scripts/classify_embeddings_nhy3.py --gpu 0
```

Salida: `result/exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc/inference/calib_roi3ch_uni_rois_reproc/classification_nhy3/`.

### Estadio PD HY1 vs HY2 — ClinicalBERT + CLIP v2 (`clinical_staging_pd_nhy12`)

Pipeline **PD-only** (109 sujetos: train 89, test 20) para clasificar **H&Y estadio 1 vs 2** con prompts clínicos neutros (sinHY, EN) + embeddings **ClinicalBERT** + alineación **CLIP** imagen–texto frente a baselines KNN (imagen y texto). Split **primario** `data_info/train.txt` + `test.txt` (no el `split` de `nhy_distribution.csv`).

**CLIP v2:** loss híbrida `L = 0.5·ClassAwareInfoNCE + 1.0·SupCon(img)`; `StandardScaler` en latentes congelados (img+txt); early stop por **menor `val_supcon_loss`** (`min_epochs=5`, `patience=20`, `seed=42`). Evaluación en dos fases: **alineación** (`metrics_alignment.csv`, retrieval R@1) vs **staging** (`metrics_staging.csv`, prototipos + KNN joint + late fusion ML en `clip/fusion/`). Referencia baseline: KNN imagen crudo k=5 **BA ≈ 0.64** en test.

| analysis_id | Parent radiológico | Salida unificada |
|-------------|-------------------|------------------|
| `clinical_staging_pd_nhy12` | `calib_roi3ch_uni_rois_reproc` (solo lectura) | `result/clinical_staging_pd/` |

**GPU:** pasos con CUDA usan **GPU 7** por defecto. **Lanzamiento:** siempre con `nohup` (logs en `logs/`).

```bash
mkdir -p logs
export CUDA_VISIBLE_DEVICES=7

# Pipeline completo (secuencial)
setsid nohup bash scripts/run_clinical_staging_pd.sh \
  > logs/clinical_staging_pd.log 2>&1 < /dev/null &

# O paso a paso:
setsid nohup python3 scripts/extract_clinical_text_embeddings.py \
  --out-root result/clinical_staging_pd --gpu 7 \
  > logs/clinical_staging_text_embeddings.log 2>&1 < /dev/null &

setsid nohup python3 scripts/classify_nhy_staging_pd.py \
  --out-root result/clinical_staging_pd \
  > logs/clinical_staging_baselines.log 2>&1 < /dev/null &

# CLIP v2 (entrenamiento + eval post-hoc staging/fusion)
setsid nohup python3 scripts/train_clinical_radiology_clip.py \
  --out-root result/clinical_staging_pd --gpu 7 \
  --lambda-infonce 0.5 --lambda-supcon 1.0 --seed 42 \
  > logs/clinical_staging_clip_v2.log 2>&1 < /dev/null &
```

Árbol de salida:

```
result/clinical_staging_pd/
  run_config.json
  clinical_pairs.csv
  prompts.csv
  text_embeddings/
  baselines/          # KNN k=5 imagen + texto; BA, recall_hy1/hy2, f1, AUC
  clip/
    training_history.csv      # val_supcon_loss (early stop), val_retrieval_r1 (diagnóstico)
    metrics_alignment.csv   # retrieval R@1 (alineación img↔texto)
    metrics_staging.csv       # Proto_text/image/avg, KNN_joint (BA, recall, f1, AUC)
    fusion/                   # late fusion: KNN, LogReg, SVM, DNN sobre [z_img; z_txt]
    checkpoints/
      projectors.pt
      feature_scalers.joblib
```

Entrada clínica: `MDS-UPDRS_Part_III_28Oct2025_estandarizado_pd_mbrs.csv`. Latentes imagen: `result/exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc/inference/calib_roi3ch_uni_rois_reproc/` (`sampled_latent_final`).

### E2 ROI reproc + spatial FiLM (`calib_roi3ch_uni_rois_reproc_film_ms`)

Ablación **única** sobre `calib_roi3ch_uni_rois_reproc`: condicionamiento MRI espacial en cada `ResnetBlock` vía mapa **h64** (`γ,β` por vóxel; resize trilinear por nivel UNet). Difusión SPECT sigue en **z18k**; concat z18+ROI sin cambios. EMA legacy (sin warmup).

| Clave registry | Parent | Delta |
|----------------|--------|-------|
| `calib_roi3ch_uni_rois_reproc_film_ms` | `calib_roi3ch_uni_rois_reproc` | `ILCLDM_FILM_MODE=spatial_ms` |

```bash
source scripts/roi_profile_env.sh

# Smoke VRAM (1 época)
ILCLDM_EPOCHS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc_film_ms --gpu <N>

# Entrenamiento
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc_film_ms --gpu <N>

# Inferencia + clasificación (métrica guía: F1/recall embedding DDIM)
python3 inference.py \
  --exp exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_film_ms \
  --run-id calib_roi3ch_uni_rois_reproc_film_ms \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri_rois_reproc/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

Si OOM en batch 8, añadir `"ILCLDM_BATCH_SIZE": "4"` en overrides del registry. UNet **scratch** (no warm-start desde parent).

### E2 ROI reproc + spatial FiLM + MRI finetune (`calib_roi3ch_uni_rois_reproc_film_ms_ft`)

Ablación **única** sobre `calib_roi3ch_uni_rois_reproc_film_ms`: el encoder MRI **deja de estar congelado** (`ILCLDM_MRI_ENCODER_MODE=finetune`). Parte del checkpoint E1 `exp_nm_s1_mri_rois_reproc` y se entrena junto al UNet (LR MRI = `ILCLDM_LR × 0.25`). AAE SPECT sigue congelado. UNet **scratch**. Mismos datos/FiLM/spatial_ms que el parent.

| Clave registry | Parent | Delta |
|----------------|--------|-------|
| `calib_roi3ch_uni_rois_reproc_film_ms_ft` | `calib_roi3ch_uni_rois_reproc_film_ms` | `ILCLDM_MRI_ENCODER_MODE=finetune` |

```bash
source scripts/roi_profile_env.sh

ILCLDM_EPOCHS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc_film_ms_ft --gpu <N>

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc_film_ms_ft --gpu <N>

# Inferencia: usar el encoder MRI finetuned guardado en result/<exp>/
python3 inference.py \
  --exp exp_nm_s2_finetune_calib_roi3ch_uni_rois_reproc_film_ms \
  --run-id calib_roi3ch_uni_rois_reproc_film_ms_ft \
  --mri-encoder-mode finetune \
  --mri-encoder-checkpoint result/exp_nm_s2_finetune_calib_roi3ch_uni_rois_reproc_film_ms/MRI_Encoder_finetuned.pth.tar \
  --save-synthetic-spect --splits train test
```

Exportar `ILCLDM_FILM_MODE=spatial_ms` en inferencia si `apply_manifest_to_config` no restaura `FILM_MODE`.

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

## 5z. Full-brain port ROI→FB (`calib_fb_clamp` / `calib_fb_v2`)

Rama `feat/fullbrain-roi-port`. Aplica los fixes validados en ROI sobre **datos full-brain** (`data/`, crop → `(160,192,160)`, latente `(40,48,40)`).

### Env nuevos (defaults)

| Variable | Default | Rol |
|----------|---------|-----|
| `ILCLDM_PRED_X0_CLAMP` | `4.0` | Clamp \|pred_x0\| en train + DDIM |
| `ILCLDM_LABEL_DROPOUT_PROB` | `0` | Token nulo si >0 (embedding +1 fila) |
| `ILCLDM_EMA_BETA` / `_START` | `0.995` / `0.9` | Warmup obligatorio de beta |
| `ILCLDM_EMA_WARMUP_STEPS` / `_STEP_START` | `0` (=auto) | Auto: max(200, 5% steps) / max(1, 5% steps) |
| `ILCLDM_ATTN_RES` | `4` | Full-brain v2: **`5` solo** (nunca `(5,10)`; OOM O(N²)) |)
| `ILCLDM_USE_FILM` | `0` | FiLM global (GAP→MLP bias); legacy si `FILM_MODE=off` |
| `ILCLDM_FILM_MODE` | `off` | `off` \| `global` \| `spatial_ms` (h64 γ/β por bloque) |
| `ILCLDM_FILM_COND_CHANNELS` | `64` | Canales del mapa MRI para spatial FiLM (encoder h64) |
| `ILCLDM_EARLY_STOP_MIN_DELTA` | `0` | Mejora mínima para resetear paciencia |
| `ILCLDM_USE_WANDB` | `0` | Logging opcional |
| `ILCLDM_INCLUDE_SVM` | `0` | SVM off (cuelga en latentes altos-D) |

### Datos

- IDs: 181 en `data_info` ↔ `data/whole_*`
- Latentes train: `data/latent_Abeta/` (regenerados con `exp_nm_s1_spect`)
- Stats: `data_info/latent_stats_latent_Abeta.json`
- Máscara: `data/mask_basal_ganglia_full.nii.gz` (symlink `mask_basal_ganglia.nii.gz`)

### Runs

```bash
# Smoke 1 época (memoria)
ILCLDM_EPOCHS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_fb_clamp --gpu 1

# Baseline clamp-only (legacy EMA/attn)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_fb_clamp --gpu 1

# Full improvements (needs ≥24GB VRAM at batch=2; A2000 12GB OOMs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/run_ablation.py calib_fb_v2 --gpu 1

# Inferencia + clasificación (sin SVM) + proyecciones 2D/3D
python3 inference.py --exp exp_nm_s2_frozen_calib_fb_clamp --run-id calib_fb_clamp \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

**VRAM:** `calib_fb_clamp` (attn@4, no FiLM) cabe en 12–24GB a `BATCH_SIZE=2`. `calib_fb_v2` (attn@5 + FiLM, 6 bloques attn) requiere **≥24GB** (4090/Titan RTX); en A2000 12GB falla OOM incluso con `expandable_segments`.

Métrica guía A/B: **AUC / balanced accuracy del latente DDIM**, no PSNR.

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
| `bash scripts/run_e1_rois.sh` | E1 SPECT + MRI en `data_rois` + stats latentes ROI |

Documentación ampliada: [DEVCONTAINER.md](DEVCONTAINER.md).
