# Informe de preparacion - Fase 4

Fecha: 2026-05-17

## Objetivo

Preparar la fase final de integracion del TFM mediante un notebook y un script reproducible que consoliden los resultados de clasificacion, segmentacion y explicabilidad Grad-CAM.

## Artefactos creados

- `scripts/build_final_analysis.py`
- `notebooks/07_final_analysis.ipynb`
- `results/final_analysis/`

## Tablas finales

- `results/final_analysis/classification_results_with_ci.csv`
- `results/final_analysis/classification_best_by_accuracy.csv`
- `results/final_analysis/classification_best_by_f1.csv`
- `results/final_analysis/classification_mcnemar_top2.csv`
- `results/final_analysis/segmentation_results.csv`
- `results/final_analysis/segmentation_best_by_dice.csv`
- `results/final_analysis/xai_gradcam_results.csv`
- `results/final_analysis/rq_summary.md`

## Figuras finales

- `results/final_analysis/figures/classification_accuracy_f1_macro.png`
- `results/final_analysis/figures/segmentation_dice_iou.png`
- `results/final_analysis/figures/xai_gradcam_alignment.png`

## Resultados clave

### Clasificacion

- Mejor CXR: `cxr_densenet121_weighted_ce`
  - Accuracy: `0.9496`
  - F1-macro: `0.9567`
  - AUC macro: `0.9931`
- Mejor CT por accuracy: `ct_resnet50_baseline`
  - Accuracy: `0.6484`
- Mejor CT por F1-macro/AUC: `ct_densenet121_baseline`
  - F1-macro: `0.4173`
  - AUC macro: `0.7229`

### Segmentacion

- Mejor CXR: `cxr_attention_unet_segmentation`
  - Dice: `0.9853`
  - IoU: `0.9715`
- Mejor CT: `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`
  - Dice: `0.5637`
  - IoU: `0.4305`

### Explicabilidad Grad-CAM

- CXR Grad-CAM vs mascara pulmonar:
  - IoU medio: `0.2255`
  - Ratio dentro de mascara: `0.3143`
- CT Grad-CAM vs mascara de infeccion:
  - Mejor IoU observado: `0.0146`
  - Pico dentro de mascara: `0.0000`

## Lectura por RQ

- **RQ1:** DenseNet-121 con weighted CE domina en CXR. En CT no hay un unico ganador: ResNet-50 gana levemente en accuracy, DenseNet-121 en F1/AUC.
- **RQ2:** CXR es mucho mas estable que CT; la severidad CT por slices 2D es la tarea mas dificil.
- **RQ3:** Grad-CAM tiene plausibilidad anatomica parcial en CXR, pero no se alinea con lesion CT.
- **RQ4:** Attention U-Net es la mejor familia en ambas modalidades, con CXR resuelto y CT aun limitado.
- **RQ5:** Weighted CE ayuda en CXR; en CT las estrategias de balanceo no superan claramente al baseline.

## Siguiente paso

Ejecutar y revisar `notebooks/07_final_analysis.ipynb`. Despues, trasladar las tablas/figuras al capitulo de resultados y usar `results/final_analysis/rq_summary.md` como base para la discusion.
