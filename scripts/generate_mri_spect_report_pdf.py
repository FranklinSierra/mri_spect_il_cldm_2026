#!/usr/bin/env python3
"""Generate PDF report: MRI vs SPECT comparison and synthesis recommendations."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = ROOT / "docs" / "reports" / "mri_spect_stats"
OUT_PDF = STATS_DIR / "INFORME_COMPARACION_MRI_SPECT.pdf"
FIG_PATH = STATS_DIR / "mri_spect_comparison.png"
REMOVED_IDS = {"3378", "4033", "41184"}


def load_stats(csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(STATS_DIR / csv_name)
    df["ID"] = df["ID"].astype(str)
    return df[~df["ID"].isin(REMOVED_IDS)].copy()


def cohort_metrics(df: pd.DataFrame) -> dict:
    def cohens_d(a, b):
        a, b = a.to_numpy(), b.to_numpy()
        return float((a.mean() - b.mean()) / ((a.var() + b.var()) / 2) ** 0.5)

    r_mean, p_mean = stats.pearsonr(df["mri_mean"], df["spect_mean"])
    r_p99, p_p99 = stats.pearsonr(df["mri_p99"], df["spect_p99"])
    ks = stats.ks_2samp(df["mri_mean"], df["spect_mean"])
    band = ((df["ratio_mean"] >= 0.8) & (df["ratio_mean"] <= 1.2)).mean() * 100
    out = {
        "n": len(df),
        "mri_mean": df["mri_mean"].mean(),
        "spect_mean": df["spect_mean"].mean(),
        "mri_std": df["mri_std"].mean(),
        "spect_std": df["spect_std"].mean(),
        "ratio_med": df["ratio_mean"].median(),
        "ratio_iqr": df["ratio_mean"].quantile(0.75) - df["ratio_mean"].quantile(0.25),
        "pct_band": band,
        "r_mean": r_mean,
        "p_mean": p_mean,
        "r_p99": r_p99,
        "mri_p99_med": df["mri_p99"].median(),
        "spect_p99_med": df["spect_p99"].median(),
        "ratio_std_med": df["ratio_std"].median(),
        "cohen_mean": cohens_d(df["mri_mean"], df["spect_mean"]),
        "cohen_std": cohens_d(df["mri_std"], df["spect_std"]),
        "ks_D": ks.statistic,
        "ks_p": ks.pvalue,
    }
    if "spect_p999" in df.columns:
        out["spect_p999_med"] = df["spect_p999"].median()
    return out


def wrap_paragraphs(text: str, width: int = 95) -> list[str]:
    lines: list[str] = []
    for para in text.strip().split("\n\n"):
        lines.extend(textwrap.wrap(para, width=width))
        lines.append("")
    return lines


def draw_text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.07, y, title, fontsize=14, fontweight="bold", va="top")
    y -= 0.04
    for line in lines:
        if y < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.95
        fig.text(0.07, y, line, fontsize=fontsize, va="top", family="sans-serif")
        y -= 0.028 if line else 0.018
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_full = load_stats("per_subject_mri_spect_stats.csv")
    df_crop = load_stats("per_subject_mri_spect_stats_cropped.csv")
    m_full = cohort_metrics(df_full)
    m_crop = cohort_metrics(df_crop)

    today = date.today().isoformat()

    cover_lines = wrap_paragraphs(
        f"""
Proyecto: IL-CLDM (cohorte fsierra, dataV2)
Fecha del informe: {today}
Cohorte analizada: {m_crop["n"]} sujetos emparejados MRI + SPECT
IDs excluidos por volumetria anomala: {", ".join(sorted(REMOVED_IDS))}

Datos: volumenes z-score por sujeto (MRI estructural y SPECT DaT).
Carpetas: dataV2/whole_MRI y dataV2/whole_Abeta.
Metadatos y splits: data_info/ (sin modificacion).

Este documento resume el analisis comparativo de estadisticos de intensidad
y recomendaciones para la sintesis MRI -> SPECT con preservacion de regiones
de alto uptake.
        """
    )

    methods_lines = wrap_paragraphs(
        """
METODOLOGIA

- IDs: nombres de archivo en dataV2/whole_MRI ({ID}.nii.gz).
- Por sujeto y modalidad: media, desviacion estandar, mediana, percentil 99
  (y p99.9 en analisis con crop), calculados sobre voxels > 0.
- Analisis sin crop: volumen completo (182 x 218 x 182).
- Analisis con crop: region IL-CLDM [10:170, 18:210, 10:170] -> 160 x 192 x 160
  (identica a dataset.py en entrenamiento).
- No se aplico crop como criterio de exclusion; el crop solo define la ROI de analisis.
- Figura de dispersion: docs/reports/mri_spect_stats/mri_spect_comparison.png
- CSV detallados: per_subject_mri_spect_stats.csv y per_subject_mri_spect_stats_cropped.csv
        """
    )

    results_table = f"""
RESULTADOS NUMERICOS (cohorte {m_crop["n"]} sujetos)

                        Volumen completo          Con crop IL-CLDM
------------------------------------------------------------------------
Media MRI (cohorte)     {m_full["mri_mean"]:.3f}                   {m_crop["mri_mean"]:.3f}
Media SPECT (cohorte)   {m_full["spect_mean"]:.3f}                   {m_crop["spect_mean"]:.3f}
Mediana ratio SPECT/MRI {m_full["ratio_med"]:.3f}                   {m_crop["ratio_med"]:.3f}
% sujetos ratio [0.8,1.2] {m_full["pct_band"]:.1f}%                  {m_crop["pct_band"]:.1f}%
Correlacion r (medias)  {m_full["r_mean"]:.3f}                   {m_crop["r_mean"]:.3f}
Correlacion r (p99)     {m_full["r_p99"]:.3f}                   {m_crop["r_p99"]:.3f}
Std media MRI           {m_full["mri_std"]:.3f}                   {m_crop["mri_std"]:.3f}
Std media SPECT         {m_full["spect_std"]:.3f}                   {m_crop["spect_std"]:.3f}
Mediana ratio std S/M   {m_full["ratio_std_med"]:.2f}                   {m_crop["ratio_std_med"]:.2f}
Mediana p99 MRI         {m_full["mri_p99_med"]:.2f}                   {m_crop["mri_p99_med"]:.2f}
Mediana p99 SPECT       {m_full["spect_p99_med"]:.2f}                   {m_crop["spect_p99_med"]:.2f}
Cohen d (medias)        {m_full["cohen_mean"]:.3f}                   {m_crop["cohen_mean"]:.3f}
Cohen d (std)           {m_full["cohen_std"]:.2f}                   {m_crop["cohen_std"]:.2f}
KS D (dist. medias)     {m_full["ks_D"]:.3f}                   {m_crop["ks_D"]:.3f}

El crop no altera sustancialmente las conclusiones: las medias globales y los
picos (p99 ~ 3.9 en SPECT vs ~ 1.5 en MRI) se mantienen.
    """

    interpretation_lines = wrap_paragraphs(
        """
INTERPRETACION

1. Escalas globales de la media (bien alineadas)
   Las medias por sujeto estan en la misma banda tras z-score por sujeto.
   Mediana del ratio SPECT/MRI ~ 0.97; ~96% de sujetos con ratio en [0.8, 1.2].
   Cohen's d en medias ~ 0.35 (efecto pequeno). No hay un desfase sistemico del
   tipo "SPECT el doble de MRI" a nivel de media global.

2. Correlacion sujeto a sujeto (baja)
   r ~ 0.10 entre mri_mean y spect_mean, y similar para p99.
   Un sujeto con media MRI alta no implica media SPECT alta: patron esperable entre
   MRI estructural y SPECT funcional. Hay alineacion de escala, no de contraste relativo.

3. Variabilidad y colas (SPECT mucho mas disperso)
   Std media SPECT ~ 1.8x MRI; mediana p99 SPECT ~ 2.5x MRI.
   El desfase principal es contraste espacial y picos de uptake, no la media.
   Con z-score, voxels en z > 3 son clinicamente relevantes pero pesan poco en L1/L2 media.

4. Sujetos excluidos
   3378 (SPECT mean=0), 4033 y 41184 (MRI mean=0) fueron eliminados de dataV2.

5. Implicaciones para entrenamiento
   - Fase 1 (autoencoder MRI): riesgo bajo tras exclusion de IDs anomalos.
   - Sintesis MRI -> SPECT: el modelo debe aprender mapping no lineal y generar
     mayor varianza y picos que el MRI; L1/L2 uniforme favorece representaciones medias.
        """
    )

    recommendations_lines = wrap_paragraphs(
        """
RECOMENDACIONES PARA PRESERVAR PICOS SPECT EN SINTESIS

Contexto codigo actual (IL-CLDM):
- Etapa 1 AAE: recon_loss = |SPECT - decoded|.mean() + GAN (main.py).
- Etapa 2 LDM: MSE sobre ruido en espacio latente, no sobre voxels SPECT.
- Normalizacion unica: z-score por volumen (sin opcion min-max); metricas usan data_range por percentiles.

A. Perdida ponderada en imagen (etapa 1 y/o decode en inferencia)
   w = 1 + alpha * relu(target - 1) o mascara en target > percentil 90.
   loss = mean(w * |target - recon|). Alpha sugerido: 2-5.

B. Perdida en colas (percentiles)
   Termino adicional L1 solo en voxels con target >= p90-p95 del volumen.
   loss = loss_global + beta * loss_tail (beta ~ 0.5-2).

C. Reforzar GAN / perceptual en regiones brillantes
   Muestrear patches con mayor probabilidad donde target > p90.
   Discriminador global + opcional en mascara de alto uptake.

D. Metricas orientadas a picos
   Ademas de PSNR/SSIM global: MAE/PSNR en mascara uptake (GT > p90),
   correlacion en ROI estriatal si hay atlas, ratio p99 sintetico / real ~ 1.

E. Etapa 2 (LDM)
   MSE en latente no garantiza picos tras decoder. Opciones:
   - Auxiliary loss en imagen cada N pasos: decode(z) vs SPECT ponderado.
   - Refinador ligero post-decode en voxels altos.

F. Muestreo y augmentacion
   Patch sampling sesgado hacia voxels con z alto; evitar clipping que comprima colas.

G. Validacion
   Clamp(0,1) eliminado (solo z-score); data_range por percentiles ya aplicado. Opcional: evaluar en mascara cerebral.

PRIORIDAD SUGERIDA
1. L1 ponderada + metricas p99/ROI en AAE.
2. Termino tail/percentile en decode (inferencia o fine-tune).
3. Auxiliary image loss en LDM (opcional).
        """
    )

    annex_lines = wrap_paragraphs(
        f"""
ANEXOS - ARCHIVOS RELACIONADOS

- Figura: {FIG_PATH.relative_to(ROOT)}
- CSV volumen completo: docs/reports/mri_spect_stats/per_subject_mri_spect_stats.csv
- CSV con crop: docs/reports/mri_spect_stats/per_subject_mri_spect_stats_cropped.csv
- PDF: docs/reports/mri_spect_stats/INFORME_COMPARACION_MRI_SPECT.pdf

Comandos de reproduccion:
  python scripts/inspect_fsierra_volumes.py --reference-data data
  python scripts/copy_fsierra_to_datav2.py --reference-data data --dest dataV2
  python scripts/evaluate_datav2.py --data-root dataV2 --reference-data data
  python scripts/generate_mri_spect_report_pdf.py
        """
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUT_PDF) as pdf:
        draw_text_page(
            pdf,
            "Informe: Comparacion MRI / SPECT (dataV2)",
            ["IL-CLDM - Analisis de intensidades y recomendaciones de sintesis", ""]
            + cover_lines,
        )
        draw_text_page(pdf, "Metodologia", methods_lines)
        draw_text_page(pdf, "Resultados", wrap_paragraphs(results_table))
        draw_text_page(pdf, "Interpretacion", interpretation_lines)
        draw_text_page(pdf, "Recomendaciones", recommendations_lines)
        draw_text_page(pdf, "Anexos", annex_lines)

        if FIG_PATH.exists():
            fig, ax = plt.subplots(figsize=(8.27, 7))
            img = plt.imread(FIG_PATH)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("Figura: comparacion MRI vs SPECT (dataV2)", fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
