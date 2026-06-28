# Indice de notebooks

Este directorio contiene el recorrido principal del TFM. Los notebooks estan ordenados por fase, no necesariamente por orden cronologico de creacion.

## Recorrido principal

| Notebook | Funcion |
| --- | --- |
| `00_eda.ipynb` | Exploracion inicial de datasets CXR y CT, conteos, ejemplos y disponibilidad de mascaras. |
| `02_cxr_classification.ipynb` | Entrenamiento de clasificadores CXR en modo full. |
| `03_ct_classification.ipynb` | Entrenamiento de clasificadores CT por slice en modo full. |
| `04_classification_results.ipynb` | Lectura y comparacion de resultados de clasificacion CXR/CT. |
| `05_segmentation.ipynb` | Entrenamiento principal de segmentacion CXR y CT. |
| `05k_ct_segmentation_visualization.ipynb` | Visualizacion cualitativa de segmentacion CT por modelo. |
| `06_explainability.ipynb` | Grad-CAM para CXR y CT, con metricas de solapamiento saliencia-mascara. |
| `08_calibration_analysis.ipynb` | Calibracion probabilistica, diagramas de fiabilidad y confianza. |
| `09_ct_study_level_analysis.ipynb` | Agregacion de predicciones CT por estudio. |
| `10_ct_informative_slice_selection.ipynb` | Seleccion de slices informativos en CT. |
| `11_ct_modern_backbone_classification.ipynb` | Experimentos CT con backbone moderno ConvNeXt-Tiny. |
| `12_ct_study_level_meta_classifier.ipynb` | Meta-clasificador CT a nivel de estudio. |
| `07_final_analysis.ipynb` | Resumen final de resultados y figuras globales. |

## Experimentos auxiliares

Los notebooks de `experiments/ct_segmentation/` contienen variantes usadas para mejorar segmentacion CT:

- Tversky ponderada.
- Entrenamiento por parches.
- Contexto mixto.
- Postprocesado.
- Exploracion 2.5D.
- Ablaciones.
- Ensembles.
- Slices negativos.
- Refinamiento de capacidad.

Se mantienen para trazabilidad experimental, pero en la memoria se recomienda describirlos como ablation study o experimentos adicionales, no como flujo principal.

## Criterio de limpieza

Se han retirado guias automaticas linea a linea y salidas embebidas pesadas para que los notebooks sean mas faciles de leer y versionar. Se conservan las explicaciones metodologicas y las celdas reproducibles.
