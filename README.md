# TFM - Clasificacion, segmentacion e interpretabilidad en imagen medica

Este repositorio contiene el flujo experimental del TFM sobre imagen medica CXR y CT relacionada con COVID-19 y patologias respiratorias.

## Estructura principal

- `notebooks/`: recorrido reproducible del TFM. Ver `notebooks/README.md`.
- `scripts/`: scripts ejecutables usados por los notebooks.
- `src/`: codigo fuente reutilizable para datos, modelos, entrenamiento y evaluacion.
- `results/`: resultados completos generados por los experimentos.
- `figures/`: figuras seleccionadas listas para insertar en la memoria.
- `deliverables/`: paquete limpio con figuras, tablas e informes finales. Ver `deliverables/README.md`.
- `docs/`: bitacora, borradores e informes de apoyo.

## Flujo recomendado

1. `notebooks/00_eda.ipynb`
2. `notebooks/02_cxr_classification.ipynb`
3. `notebooks/03_ct_classification.ipynb`
4. `notebooks/04_classification_results.ipynb`
5. `notebooks/05_segmentation.ipynb`
6. `notebooks/05k_ct_segmentation_visualization.ipynb`
7. `notebooks/06_explainability.ipynb`
8. `notebooks/08_calibration_analysis.ipynb`
9. `notebooks/09_ct_study_level_analysis.ipynb`
10. `notebooks/10_ct_informative_slice_selection.ipynb`
11. `notebooks/11_ct_modern_backbone_classification.ipynb`
12. `notebooks/12_ct_study_level_meta_classifier.ipynb`
13. `notebooks/07_final_analysis.ipynb`

Los notebooks en `notebooks/experiments/` son experimentos auxiliares, principalmente variantes de segmentacion CT. Se conservan por trazabilidad, pero no forman parte del recorrido principal de lectura.

## Nota de reproducibilidad

Los experimentos estan configurados en modo `full`. Los resultados ya generados se encuentran en `results/` y las tablas/figuras principales se han copiado a `deliverables/`.
