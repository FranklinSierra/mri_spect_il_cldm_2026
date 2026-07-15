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

Patches ya recortados (bbox `mask_basal_ganglia` + 15 vox). **No** se aplica el crop IL-CLDM `[10:170,…]` en carga (`ILCLDM_SKIP_CROP=1`). Shape en disco: `(105, 102, 102)`; el pipeline añade padding cero a `(108, 104, 104)` para que AAE/UNet preserven tamaño; grid latente `(27, 26, 26)`.

### Preflight ROI

```bash
python3 scripts/preflight_data_rois.py
python3 scripts/aggregate_roi_unet_mask.py   # una vez: data_rois/roi_mask_unet.nii.gz
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
| `ILCLDM_ROI_VOLUME_SHAPE` | `105,102,102` |

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
| `bash scripts/run_e1_rois.sh` | E1 SPECT + MRI en `data_rois` + stats latentes ROI |

Documentación ampliada: [DEVCONTAINER.md](DEVCONTAINER.md).
