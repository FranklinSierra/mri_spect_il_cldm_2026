# Comparación del modelamiento de difusión: IL-CLDM vs PASTA

**Fecha de generación:** 2026-09-29
**Alcance:** análisis comparativo de la parte generativa basada en difusión entre `IL-CLDM` (`/workspace/IL-CLDM`) y `PASTA` (`/workspace/pasta-pd-ref/PASTA_CON_CLINICA`), con el objetivo de identificar qué hace PASTA que permite obtener síntesis coherentes y qué está impidiendo lo mismo en IL-CLDM.
**Referencia cruzada:** ver `docs/REPORTE_PROYECTO.md` §5 para la evidencia empírica del fallo (síntesis MRI→SPECT con PSNR ≈ 2.94, SSIM ≈ 0.048, salida tipo "blob").

---

## 1. Resumen ejecutivo

La diferencia decisiva no es un bug aislado, son tres decisiones de diseño acopladas:

| Aspecto | PASTA (funciona) | IL-CLDM (no funciona) |
|---|---|---|
| Espacio de difusión | Imagen, acotada a `[-1, 1]` | Latente AAE sin acotar, z-score escalar global |
| Objetivo del modelo | `START_X` (predice x₀) + L1 + SSIM | `EPSILON` (predice ruido) + MSE |
| Proyección por paso | `clip_denoised=True` → clamp a `[-1,1]` en cada paso | `PRED_X0_CLAMP=10.0`, tan ancho que casi no recorta |
| Condicionamiento | UNet gemelo, features multiescala vía AdaGN en **cada bloque** | 1 canal concatenado al input |
| Entrenamiento del condicionador | End-to-end, con pérdida de reconstrucción MRI→PET | Encoder MRI **congelado**, entrenado solo para reconstruir MRI |

En esa combinación, cualquier error del modelo de IL-CLDM se amplifica por $1/\sqrt{\bar\alpha_t}$ (hasta 64.171×) y no hay nada que ancle la salida a la anatomía del sujeto. El propio `config.py:32-48` ya documenta el síntoma exacto sin nombrar la causa.

**Cambio de mayor retorno:** pasar a predecir x₀ en lugar de ε. Elimina la amplificación y permite clampear a un rango válido en cada paso. **No** crea información que no esté en el condicionamiento, así que debe combinarse con §4.

---

## 2. Causa 1 — Parametrización ε con MSE en lugar de x₀

### 2.1 El mecanismo

La reconstrucción de la imagen limpia desde el ruido predicho es:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}}
$$

Si el error en el ruido es $\Delta\varepsilon = \hat\varepsilon - \varepsilon$, el error en la imagen limpia es:

$$
\Delta x_0 = -\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\,\Delta\varepsilon
$$

Ese factor es el inverso de la raíz del SNR y **explota** al final del schedule.

### 2.2 Números medidos en IL-CLDM

`config.py:36-38` documenta un encogimiento constante de $\langle\hat\varepsilon,\varepsilon\rangle/\langle\varepsilon,\varepsilon\rangle = 0.71$, es decir, un error residual del 29%:

| t | Amplificación $1/\sqrt{\bar\alpha_t}$ | Error resultante en x₀ |
|---|---|---|
| 990 | 71.3× | 0.29 × 71.3 ≈ **20.7** |
| 999 | 64.171× | 0.29 × 64.171 ≈ **18.600** |

La desviación estándar del latente calibrado es 1 y la varianza del objetivo es 3.90. O sea, **el error supera al señal por un factor de ~5 a ~9.500**.

### 2.3 Por qué la pérdida se ve bien

Si $\hat\varepsilon = 0.71\varepsilon$, la MSE del ruido es $0.29^2 = 0.084$, contra una línea base de 1.0 (predecir cero). La curva de pérdida marca 0.08 y parece excelente.

Peor: existe una **solución trivial** que la pérdida en ε premia. A t alto, $x_t \approx \sqrt{1-\bar\alpha_t}\,\varepsilon$, así que el modelo puede leer el ruido del propio input:

$$
\hat\varepsilon = \frac{x_t}{\sqrt{1-\bar\alpha_t}} \quad\Longrightarrow\quad \hat{x}_0 = 0 \ \text{exactamente}
$$

MSE casi cero y cero información sobre x₀. **Predecir ε desde x_t es fácil; predecir x₀ es lo difícil.** La pérdida en ε no distingue entre ambas cosas, y por eso el encogimiento de 0.71 es el comportamiento clásico de una regresión MSE sub-determinada: cuando el modelo no puede resolver *dónde* está la señal, la respuesta óptima bajo MSE es la media condicional, que es una versión atenuada del objetivo.

### 2.4 Cómo lo hace PASTA

`src/config/pasta_mri2pet_parkinson_mid_train.yaml`:

```yaml
objective: START_X
loss_type: l1
```

Y añade SSIM automáticamente cuando el objetivo es x₀ (`gaussian_diffusion.py:895-897`):

```python
if self.model_mean_type == ModelMeanType.START_X:
    terms['ssim_loss'] = torch.mean(torch.stack([(1-ssim(...)) for i in range(target.shape[1])], dim=0))
    terms['loss'] += terms['ssim_loss']
```

Predeciendo x₀ directamente, un error del 29% es un error del 29% en la imagen. No se multiplica por nada.

### 2.5 Qué esperar del cambio

| Sí arregla | No arregla |
|---|---|
| La amplificación $1/\sqrt{\bar\alpha}$ | La falta de información en el condicionamiento |
| Permite clampear a rango válido por paso | El poder discriminativo control vs Parkinson |
| Permite volver a `SAMPLING_T_MAX=999` | La calidad del decoder AAE |
| PSNR/SSIM deberían subir de forma clara | |

El resultado visual pasa de ruido a **una imagen borrosa pero anatómicamente plausible**: un SPECT promedio bien formado. Para métricas de imagen es una mejora grande; para clasificación solo mejora si se ataca también §4.

---

## 3. Causa 2 — Dominio acotado y proyección por paso

PASTA normaliza a `[0,1]` → `[-1,1]` y **proyecta `pred_xstart` de vuelta al rango válido en cada paso** del DDIM (`gaussian_diffusion.py:283-288`):

```python
def process_xstart(x):
    if denoised_fn is not None:
        x = denoised_fn(x)
    if clip_denoised:
        return x.clamp(-1, 1)
    return x
```

IL-CLDM difunde en el latente AAE z-scoreado con **un escalar global** (`dataset.py:167-189`, `compute_latent_stats`), y ese latente tiene cola izquierda hasta −13σ (documentado en `config.py:22-24`). Con `PRED_X0_CLAMP=10.0` el recorte es prácticamente inoperante.

Sin esa proyección, los errores del paso *t* se realimentan al paso *t−1* y el muestreo diverge. Nótese la advertencia que ya está en `config.py:45-48`: bajar `t_max` compensa el síntoma pero no arregla la causa, y apretar el clamp "es lo que ocultó la divergencia durante meses".

### 3.1 El schedule de ruido NO es la causa

Ambos proyectos usan coseno. Calculé las colas de los dos:

| | β máximo | ᾱ en t=T−1 | Amplificación 1/√ᾱ |
|---|---|---|---|
| IL-CLDM | 0.9999 | 2.43e−10 | 64.171× |
| PASTA | 0.9990 | 2.43e−09 | 20.291× |

La diferencia (por el `max_beta=0.999` de PASTA en `diff_utils.py:21` frente al clip en 0.9999 de `main.py:191`) es de un orden de magnitud, o sea, irrelevante. **PASTA arranca su DDIM en t=999 con esencialmente la misma cola brutal y funciona igual.** Esta es la evidencia más fuerte de que ni el schedule ni `SAMPLING_T_MAX` son la causa raíz: PASTA sobrevive porque predice x₀ y lo clampea, no porque su schedule sea más suave.

---

## 4. Causa 3 — El condicionador de PASTA es un regresor MRI→PET completo

Este es el corazón de PASTA y la diferencia arquitectónica más profunda.

### 4.1 Inyección multiescala vía AdaGN

PASTA corre un **UNet gemelo sobre la MRI** con `return_features=True` y le inyecta una feature distinta a cada bloque del UNet de difusión (`unet.py:942-947`):

```python
if isinstance(cond, List):
    len_cond = len(cond) - 1
    original_input_cond = cond[0]
    encoder_cond = cond[1 : len_cond//2 + 1]
    mid_cond = cond[len_cond//2 + 1]
    decoder_cond = cond[len_cond//2 + 2:]
```

Cada bloque recibe su `encoder_cond[num]` / `mid_cond` / `decoder_cond[num]` y lo aplica como scale-shift después del GroupNorm (`cond_apply_method: 'ada'`, `unet.py:440-445`).

### 4.2 El condicionador se entrena con pérdida de reconstrucción

`gaussian_diffusion.py:879-893`:

```python
if recon_cond is not None:
    # MRI -> PET as the predefined task in the conditioner arm
    input_cond = x_start if self.condition else None
    if self.reconstructed_loss == 'l1':
        terms['recon_loss'] = mean_flat(abs(input_cond - recon_cond) * loss_weight_mask)
    terms['loss'] += self.recon_weight * terms['recon_loss']
```

Con `recon_weight: 0.1`. **El brazo condicional de PASTA por sí solo ya predice el PET desde la MRI**, así que la salida es anatómicamente coherente incluso si la parte de difusión aporta poco.

### 4.3 Qué hace IL-CLDM

El condicionamiento es un **único canal** concatenado al input (`diffusion_features.py:31-35`):

```python
def unet_in_channels(space=None, use_roi=None):
    c = diffusion_channels(space)   # 1 para z18k
    return 2 * c + int(bool(use_roi))
```

Y proviene de un encoder MRI **congelado** entrenado únicamente para reconstruir MRI (`MRI_ENCODER_MODE: frozen` en casi todo `experiments/registry.json`). Sus features no tienen ninguna presión para ser predictivas del SPECT.

**Consecuencia:** si la difusión falla, no hay nada que sostenga la salida. En PASTA hay un piso de calidad garantizado; en IL-CLDM no.

---

## 5. Causa 4 — Difusión latente sobre un autoencoder sin regularizar

PASTA no usa autoencoder: difunde directamente en píxel, lo que elimina toda esta clase de fallos.

IL-CLDM difunde en un latente de **1 canal** (`config.py:8`, `latent_dim = 1`), de 160×192×160 a 40×48×40 (64× de compresión), **sin KL ni VQ**. El `Discriminator` del AAE opera sobre la imagen, no sobre el código latente, así que pese al nombre "AAE" no hay regularización adversarial del prior. Es un objetivo de difusión hostil: no hay garantía de que puntos intermedios de ese espacio decodifiquen a algo válido.

### 5.1 Bug: fuga de gradiente del discriminador al generador

`main.py:408-428`:

```python
decoded_Abeta = model(Abeta)
disc_real = disc(Abeta)
disc_fake = disc(decoded_Abeta)      # <-- decoded_Abeta NO está detachado
...
opt_model.zero_grad()
loss.backward(retain_graph=True)     # gradientes correctos en el generador
opt_disc.zero_grad()                 # solo limpia el discriminador
disc_loss.backward()                 # gradientes en disc Y TAMBIÉN en el generador
opt_model.step()                     # aplica correctos + espurios
```

`disc_loss.backward()` acumula gradientes en el generador antes de `opt_model.step()`. El AAE recibe simultáneamente la señal "haz que la reconstrucción parezca real" (`g_loss = -mean(disc_fake)`) y "haz que parezca falsa" (`d_loss_fake = mean(relu(1 + disc_fake))`). El término adversarial queda contaminado y **todo el stage 2 hereda ese latente degradado**.

Arreglo: `disc_fake = disc(decoded_Abeta.detach())` para la pérdida del discriminador, con un forward separado sin detach para `g_loss`; o reordenar para que `opt_model.step()` ocurra antes de `disc_loss.backward()`.

---

## 6. Causa 5 — La atención está prácticamente desactivada

### 6.1 De dónde salen las resoluciones

`now_res` es un contador en el constructor del UNet que sigue el tamaño espacial del mapa de features (`model.py:402-417`):

```python
now_res = image_size            # = get_latent_shape()[0] = 40
for ind in range(num_mults):    # num_mults = 4
    use_attn = (now_res in attn_res)
    ...
    if not is_last:
        now_res = now_res // 2
```

Con 4 niveles hay 3 downsamples: **40 → 20 → 10 → 5**.

### 6.2 Por qué no coincide

`ATTN_RES` es la lista de resoluciones donde poner atención, y la condición es `now_res in attn_res`. Con el default `(4,)`, el 4 **no está** en {40, 20, 10, 5}, así que la condición es falsa en todos los niveles. La única atención que queda es la forzada a mano en el cuello (`model.py:419-422`):

```python
self.mid = nn.ModuleList([
    _res_block(pre_channel, pre_channel, use_attn=True),   # <-- la única
    _res_block(pre_channel, pre_channel, use_attn=False),
])
```

Verificado instanciando el modelo: **1 bloque de atención** sobre ~30 bloques, en un UNet de 204.7M parámetros.

El origen del 4 es que este patrón viene de repos donde `image_size` es potencia de dos (32→16→8→4), donde "4" significa "el cuello de botella". Como 40 no es potencia de dos, la secuencia nunca aterriza en 4.

### 6.3 Comparación y arreglo

PASTA usa `attention_resolutions: [16, 8, 4]` con 4 cabezas sobre 192×224 en 2D, así que tiene atención en muchos bloques.

`ATTN_RES=5` **sí** funciona en IL-CLDM y da 6 bloques (2 en el encoder del nivel más profundo, 1 en el mid, 3 en el decoder). Es lo que recomienda el `RUNBOOK`. Cuidado con subir más: `SelfAttention` (`model.py:267-296`) es atención densa global en 3D, O(N²) con N = D·H·W:

| Resolución | Tokens | Entradas de la matriz de atención |
|---|---|---|
| 5 | 5×6×5 = 150 | 22.500 (trivial) |
| 10 | 10×12×10 = 1.200 | 1.44M (viable) |
| 20 | 20×24×20 = 9.600 | 92M por cabeza y muestra (prohibitivo) |

---

## 7. Causa 6 — Desajuste train/inferencia en el embedding de etiqueta

En entrenamiento se pasa la etiqueta y se suma al embedding temporal (`model.py:509-511`):

```python
label = self._resolve_label(label, x.shape[0], x.device)
if label is not None and t is not None:
    t = t + self.label_emb(label)
```

Con `USE_TRAIN_LABEL=1` (default), `train_label` siempre llega en `main.py:799`. Pero en muestreo **nunca se pasa** (`main.py:319-324`):

```python
sampled_latent = diffusion.sample(
    lambda x, y, t: ema_Unet(
        x, y, t, mri_film=film, precomputed_condition=True   # <-- label queda en None
    ),
    MRI_cond,
)
```

`inference.py:949-956` hace lo mismo. Como `LABEL_DROPOUT_PROB=0`, `null_label_id` es `None`, así que `_resolve_label` devuelve `None` y no se suma nada. **El modelo se entrena con un desplazamiento del embedding temporal que en generación desaparece.**

Arreglo: o pasar la etiqueta en muestreo, o activar `LABEL_DROPOUT_PROB>0` para que exista token nulo y entrenar con dropout de etiqueta (que además habilita classifier-free guidance).

---

## 8. Causa 7 — Régimen de datos y optimización

| | PASTA | IL-CLDM |
|---|---|---|
| Representación | 2.5D: slabs de 15 cortes como 15 canales, UNet **2D** 192×224 | Volumen 3D completo, UNet **3D** 40×48×40 latente |
| Muestras por sujeto | 16 slabs solapados | 1 volumen |
| Sujetos | — | 147 train / 34 val / 34 test |
| Parámetros del modelo | `model_channels=64`, mults [1,2,3,4], 2D | **204.7M**, 3D |
| Pasos | 80.000 | ~19 pasos/época |
| Clipping de gradiente | `clip_grad_norm_(..., 1.0)` | **ninguno** |
| Scheduler de LR | `CosineAnnealingLR` | **ninguno** |
| EMA | 0.999 | 0.995 |
| Augmentación | flip/affine desactivados | ninguna en stage 2 |

### 8.1 Qué es el 2.5D exactamente

Dos parámetros distintos:

**`input_slice_channel: 15` → `output_dim=15`** es el 2.5D. Cada muestra es un paquete de 15 cortes axiales contiguos apilados como los **15 canales** de una imagen 2D de 192×224, o sea un tensor `(15, 192, 224)`. La red es un UNet 2D (`dims: 2`) con `in_channels=15` y `out_channels=15`: denoisea los 15 cortes a la vez, viendo contexto 3D local (±7 cortes) pagando costo de convoluciones 2D. El slab se arma con `half_dim = (output_dim-1)//2 = 7` en `data_utils.py:187`.

**`num_slices: 16`** es cuántas *posiciones de centro* se muestrean por sujeto. El bucle es `range(-8, 8)` como offset al corte medio (`dataset.py:481-483`). Si el eje axial tiene 182 cortes (según `eval_resolution`), el medio es `182//2+1 = 92` y los centros son 84…99:

| Centro | Cortes del slab |
|---|---|
| 84 | 77 … 91 |
| 85 | 78 … 92 |
| … | … |
| 99 | 92 … 106 |

Dos observaciones importantes:

1. **Se solapan muchísimo.** Centros consecutivos comparten 14 de 15 cortes. No son 16 muestras independientes, son 16 ventanas deslizantes de un paso; funcionan como aumentación por traslación axial.
2. **Solo cubren el centro del cerebro.** La unión va del corte 77 al 106, o sea 30 de 182 (~16%). PASTA con esta configuración **no entrena sobre el cerebro completo**, solo la franja central, que es donde está el estriado y por tanto la señal del DAT-SPECT. `num_slices: 'all'` sería la alternativa para cobertura total.

La ventaja real de PASTA no es "miles de muestras ricas", es que **el problema es mucho más pequeño** (una franja 2D de 15 canales en vez de un volumen 3D completo) con la misma cantidad de sujetos.

### 8.2 `clip_grad_norm_` y `PRED_X0_CLAMP` son cosas distintas

No hay clipping de gradiente en IL-CLDM (verificado: cero coincidencias de `clip_grad`, `clip_grad_norm_`, `clip_grad_value_` en todo el repo). `main.py:880-882` va directo de `backward()` a `step()`.

Lo que existe con rango ±10 es otra cosa (`config.py:25`):

| | `PRED_X0_CLAMP = 10.0` | `clip_grad_norm_(params, 1.0)` |
|---|---|---|
| Qué acota | El **valor del latente limpio predicho** durante el muestreo | La **norma de la actualización de pesos** en entrenamiento |
| Cuándo actúa | Inferencia y cálculo de `x0_hat` | Entre `backward()` y `step()` |
| Equivalente en PASTA | `clip_denoised=True` | `trainer.py:412` |

`clip_grad_norm_` calcula la norma L2 global de todos los gradientes concatenados en un solo vector, $\|g\| = \sqrt{\sum_p \|g_p\|^2}$, y si supera el umbral multiplica **todos** los gradientes por $1.0/\|g\|$. Preserva la dirección del paso y solo acorta su magnitud (a diferencia de `clip_grad_value_`, que clampea elemento por elemento y sí deforma la dirección). En difusión importa porque la magnitud de la pérdida varía enormemente según el *t* muestreado: un batch en t muy alto puede producir un gradiente órdenes de magnitud mayor que el típico, y con Adam el daño persiste cientos de iteraciones porque contamina el segundo momento.

> **Nota sobre PASTA:** su `self.opt.zero_grad()` está **dentro** del bucle de acumulación (`trainer.py:347-349`), así que la acumulación no acumula: al llegar a `opt.step()` solo sobrevive el gradiente del último micro-batch, dividido por 4. Su configuración efectiva es batch 1 con LR 4× menor, no batch efectivo de 4. Tenerlo en cuenta si se copia ese trainer como referencia.

---

## 9. Causa 8 — Sin ponderación ROI en la pérdida de difusión

PASTA aplica `loss_weight_mask` (×10 dentro del ROI de Parkinson) **dentro del término de difusión**, en ambas ramas de pérdida (`gaussian_diffusion.py:863` para MSE y `:869` para MAE, que es la activa con `loss_type: l1`; máscara construida en `dataset.py:615-616`):

```python
loss_weight_mask = np.ones_like(roi_mask, dtype=np.float32)
loss_weight_mask[roi_mask > 0.5] = 10.0
```

```python
terms["mae"] = mean_flat(abs(target - model_output) * loss_weight_mask)
```

IL-CLDM solo usa la máscara ROI en la **pérdida auxiliar de imagen**, y de forma intermitente (`AUX_IMAGE_EVERY_N=10`). El término principal de difusión (`main.py:829`) trata todos los vóxeles igual, así que el ~99% del volumen sin señal diluye el gradiente del estriado.

---

## 10. Plan de cambios priorizado

### Paso 0 — Validar el diagnóstico sin reescribir nada (horas)

Ya existe la pérdida en espacio x₀, solo está apagada (`main.py:841-845`, `config.py:102-103`):

```python
if use_latent_loss:
    latent_loss = F.l1_loss(x0_hat, latent_Abeta)
    loss = loss + config.LATENT_LOSS_LAMBDA * latent_loss
```

```bash
ILCLDM_LATENT_LOSS=1 ILCLDM_LATENT_LOSS_LAMBDA=5.0   # el default 0.5 es demasiado bajo
```

**Criterio de éxito:** que el MSE latente de validación baje de **3.90** (la varianza del objetivo, es decir, el desempeño de "predecir la media"). Si no baja de ahí, el problema es de condicionamiento y no de parametrización.

### Paso 1 — Parametrización x₀ (el cambio de mayor retorno)

1. Cambiar el objetivo a predecir x₀ con L1, no ε con MSE.
2. Calibrar `PRED_X0_CLAMP_LO` / `PRED_X0_CLAMP_HI` con percentiles reales del latente de train (no ±10), y aplicarlo en **cada** paso del DDIM.
3. Restaurar `SAMPLING_T_MAX=999`.

### Paso 2 — Condicionamiento (lo que da coherencia anatómica)

1. Descongelar el encoder MRI: `ILCLDM_MRI_ENCODER_MODE=finetune` (ya implementado).
2. Añadir pérdida de reconstrucción al brazo condicional, análoga al `recon_weight: 0.1` de PASTA, para forzar que las features MRI sean predictivas del SPECT.
3. Activar `ILCLDM_FILM_MODE=spatial_ms` para inyección multiescala en lugar de solo concatenación.

### Paso 3 — Higiene de entrenamiento

1. Arreglar la fuga de gradiente del AAE (§5.1). Limita el techo de todo lo demás.
2. Añadir `clip_grad_norm_(..., 1.0)` y un `CosineAnnealingLR`.
3. `ILCLDM_ATTN_RES=5` para tener 6 bloques de atención en lugar de 1.
4. Resolver el desajuste del `label_emb` (§7).
5. Llevar la ponderación ROI al término principal de difusión (§9).

### Alternativa estructural

Si el objetivo es replicar la calidad de PASTA y no defender la arquitectura latente, la vía corta es **abandonar el AAE y difundir en imagen 2.5D** como PASTA. Eso resuelve de golpe las causas 2, 4 y 7, a costa de renunciar al espacio latente como representación para la tarea de clasificación.

---

## 11. Índice de referencias de código

### IL-CLDM

| Tema | Ubicación |
|---|---|
| Clase `Diffusion` (schedule, noising, DDIM) | `main.py:149-273` |
| Pérdida de difusión (ε + MSE) | `main.py:829` |
| Pérdida en x₀ (desactivada) | `main.py:841-845` |
| Fuga de gradiente del AAE | `main.py:408-428` |
| Muestreo sin etiqueta | `main.py:319-324`, `inference.py:949-956` |
| Sin clipping de gradiente | `main.py:880-882` |
| UNet: resoluciones y atención | `model.py:402-422` |
| `SelfAttention` densa 3D | `model.py:267-296` |
| Suma del `label_emb` | `model.py:509-511` |
| Condicionamiento de 1 canal | `code/diffusion_features.py:31-35` |
| Diagnóstico previo del shrinkage | `config.py:32-48` |
| `PRED_X0_CLAMP` | `config.py:25-29` |
| Calibración escalar del latente | `dataset.py:167-189` |

### PASTA

| Tema | Ubicación |
|---|---|
| `training_losses` (objetivo, recon, SSIM) | `src/diffusion/gaussian_diffusion.py:757-900` |
| `clip_denoised` / `process_xstart` | `src/diffusion/gaussian_diffusion.py:283-288` |
| Normalización a `[-1,1]` | `src/diffusion/diff_utils.py:64-68` |
| Schedule coseno con `max_beta=0.999` | `src/diffusion/diff_utils.py:5-22` |
| Inyección multiescala del condicionador | `src/model/unet.py:942-947`, `:988-1032` |
| AdaGN scale-shift | `src/model/unet.py:440-445` |
| Clipping de gradiente | `src/trainer/trainer.py:412` |
| `zero_grad` mal ubicado | `src/trainer/trainer.py:347-349` |
| Construcción del slab 2.5D | `src/utils/data_utils.py:182-220` |
| Muestreo de 16 centros | `src/datasets/dataset.py:481-483` |
| Máscara de ponderación ROI | `src/datasets/dataset.py:615-616` |
| Configuración activa | `src/config/pasta_mri2pet_parkinson_mid_train.yaml` |

---

## 12. Qué no entra en la corrida calibrada

La clave `calib_roi3ch_uni_rois_reproc_x0_minmax_kl_calib` solo enciende la calibración del latente minmax+KL (`data_info/latent_stats_latent_Abeta_minmax_kl.json`), deja el clamp en ±4 y corrige las métricas de validación. Reusa los AAE congelados. No se lanza sola desde este documento.

Mediano plazo, todavía fase 2 con esos mismos AAE, después de ver el efecto de la calibración:

- Más épocas. El corte anterior fue a las 100, con unos 3 700 pasos.
- `clip_grad_norm_(..., 1.0)` entre `backward()` y `step()`.
- `CosineAnnealingLR` en el optimizador de la UNet.

Experimento completo aparte, no un flag de esta difusión: subir `KL_LAMBDA` de `1e-3` a `1e-1`. Eso reentrena los AAE y obliga a regenerar los latentes.

Largo plazo, fuera del proceso de difusión:

- Descongelar el encoder MRI y añadir una pérdida de reconstrucción en el brazo condicional, análoga al `recon_weight: 0.1` de PASTA.
- `FILM_MODE=spatial_ms` en lugar de concatenar un solo canal.
- `LABEL_DROPOUT_PROB>0`, o dejar de sumar `label_emb` solo en entrenamiento. Hoy el muestreo no recibe la etiqueta.
- `ATTN_RES` acorde al latente 24×28×29. Con cuatro niveles el cuello cae en 3. `ATTN_RES=4` o `5` no añade bloques.
