# Plan R4: SSIM en imagen sobre baseline minmax_kl

**Estado:** código y registry listos; entrenamiento/inferencia manual pendientes  
**Padre:** `calib_roi3ch_uni_rois_reproc_x0_minmax_kl`  
**Fuente operativa de esta corrida.** El plan global (`plan-ablation-pasta-diffusion.md`) apunta aquí para R4.

## Motivación

Tras R3 (`DIFFUSION_ROI_WEIGHT=10`), el ROI en L1 latente no aportó lo suficiente frente al baseline. El baseline **no** incluye SSIM en la loss (solo L1 latente + aux L1 imagen intermitente; SSIM es métrica de val/test).

PASTA, con objetivo `START_X`, suma `(1−SSIM)` en cada step sobre la predicción de imagen ([`gaussian_diffusion.py`](https://github.com/ai-med/PASTA/blob/main/src/diffusion/gaussian_diffusion.py)). Este experimento cierra la comparación de **pérdida de difusión** (x0 + L1 + SSIM imagen). Condicionamiento MRI y label dropout van después de congelar este stack.

## Referencia PASTA

- Código: `loss += mean_i (1 − ssim(target_i, pred_i))` cuando `model_mean_type == START_X`
- **λ efectiva = 1.0** (sin hiperparámetro adicional)
- Cada step de entrenamiento
- Dominio imagen (PASTA difunde en imagen; IL-CLDM decodea latente → SPECT)

Cosine / grad clip existen en PASTA pero **no** entran en esta clave (R2 queda pendiente).

## Diseño del experimento

| Campo | Valor |
| --- | --- |
| Registry key | `calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim` |
| Exp | `exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim` |
| Parent | `calib_roi3ch_uni_rois_reproc_x0_minmax_kl` |
| Base | `base_calib_roi3ch_uni_rois_reproc_minmax_kl` (mismos AAE/latentes) |
| Único override de loss | `ILCLDM_DIFFUSION_SSIM_LAMBDA=1.0` |
| Mantener | `AUX_IMAGE_LOSS` como en baseline (L1 imagen cada N) |
| No incluir | `DIFFUSION_ROI_WEIGHT`, cosine, KL1e-1, label dropout |

### Loss fase 2 (cada step)

```
L = L1(x0_hat, z_spect)                          # primary, latente
  + λ_aux * L_aux_image                          # solo si toca EVERY_N / high-t (igual baseline)
  + λ_ssim * (1 − SSIM(decode(x0_hat), SPECT_gt))  # NUEVO, cada step; λ_ssim=1.0
```

Decoder AAE **congelado**; gradiente del SSIM fluye a la UNet vía `x0_hat`. Implementación: [`losses/ssim.py`](../losses/ssim.py), flag `ILCLDM_DIFFUSION_SSIM_LAMBDA` (default `0`).

## Ejecución (manual)

```bash
python3 scripts/run_ablation.py calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim --gpu N

python3 inference.py \
  --exp exp_nm_s2_frozen_calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim \
  --run-id calib_roi3ch_uni_rois_reproc_x0_minmax_kl_ssim \
  --mri-encoder-mode frozen \
  --mri-encoder-checkpoint result/exp_nm_s1_mri_rois_reproc_minmax_kl/phase1_mri/checkpoints/MRI_AAE.pth.tar \
  --save-synthetic-spect --splits train test
```

## Criterio de éxito vs padre

- PSNR/SSIM (y ROI) test no peores que baseline; idealmente SSIM/ROI suben.
- `latent_MSE` val no colapsar de forma patológica.
- Clasificación embedding: reportar Acc, BA, Recall, F1, AUC; no exigir ganar BA.
- DDIM estable.

## Después de este experimento

Si se acepta como nuevo baseline de difusión:

1. (Opcional) R2 cosine ± grad clip — resto de optimización PASTA.
2. R5 condicionamiento MRI.
3. R6 label dropout.

**R7** del plan global (`STRUCTURE_LOSS_LAMBDA=1.0`) queda **redundante** con λ=1.0 aquí si el término es el mismo `(1−SSIM)` en SPECT; no apilar ambos.
