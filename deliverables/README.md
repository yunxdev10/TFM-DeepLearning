# Paquete de entregables del TFM

Esta carpeta resume los artefactos mas utiles para redactar y revisar la memoria.

## Carpetas

- `figures/`: figuras seleccionadas listas para insertar en LaTeX/Word.
- `tables/`: CSV principales para construir tablas de resultados.
- `reports/`: borradores, bitacora e informes de apoyo.

## Figuras recomendadas

| Figura | Uso recomendado |
| --- | --- |
| `Matrices CXR normalizadas.png` | Resultados de clasificacion CXR. |
| `ct_all_confusion_matrices_grid_normalized.png` | Resultados CT por slice. |
| `ct_study_level_best_confusion_matrix.png` | Evaluacion CT por estudio con agregacion original. |
| `ct_study_level_meta_test_metrics.png` | Comparacion de agregadores y meta-clasificador CT. |
| `ct_study_level_meta_best_confusion_matrix.png` | Matriz de confusion del mejor metodo CT final. |
| `cxr_attention_unet_segmentation_full_qualitative_grid.png` | Segmentacion pulmonar CXR cualitativa. |
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full_qualitative_grid.png` | Segmentacion CT cualitativa del mejor modelo. |
| `ct_mask_area_examples.png` | Auditoria de tamano de mascaras CT. |
| `xai_gradcam_alignment.png` | Resumen de Grad-CAM frente a mascaras. |
| `reliability_diagrams_selected_models.png` | Diagramas de fiabilidad/calibracion. |
| `confidence_histograms_selected_models.png` | Histogramas de confianza. |

## Tablas recomendadas

| Tabla | Uso recomendado |
| --- | --- |
| `classification_results_with_ci.csv` | Comparacion global de clasificadores. |
| `ct_study_level_metrics.csv` | Evaluacion CT por estudio. |
| `ct_informative_slice_metrics.csv` | Seleccion de slices informativos y ConvNeXt. |
| `ct_study_level_meta_metrics.csv` | Meta-clasificador CT por estudio. |
| `segmentation_results.csv` | Resultados de segmentacion. |
| `xai_gradcam_results.csv` | Resultados Grad-CAM. |
| `calibration_metrics.csv` | Calibracion probabilistica. |
| `ct_mask_area_audit.csv` | Auditoria de mascaras CT. |

## Nota

La carpeta `results/` conserva todos los resultados intermedios. Esta carpeta contiene una seleccion limpia para lectura y entrega.
