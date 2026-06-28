# Scripts de ejecucion y analisis

Esta carpeta contiene scripts ejecutables que producen resultados, tablas y figuras a partir del codigo de `src`.

## Scripts principales

- `run_phase1_full.py`: entrena los experimentos principales de clasificacion y segmentacion.
- `build_final_analysis.py`: consolida resultados finales para tablas y graficas resumen.
- `generate_xai_explanations.py`: genera explicaciones Grad-CAM y metricas de alineamiento.
- `build_calibration_analysis.py`: calcula metricas y figuras de calibracion probabilistica.
- `build_ct_study_level_analysis.py`: agrega predicciones CT por estudio.
- `run_ct_informative_slice_experiments.py`: entrena experimentos CT con seleccion de slices informativos.
- `build_ct_informative_slice_analysis.py`: resume los experimentos de slices informativos.
- `run_ct_modern_backbone_experiments.py`: ejecuta experimentos CT con backbone moderno.
- `run_ct_study_level_meta_classifier.py`: entrena y evalua el metaclasificador CT por estudio.
- `generate_ct_segmentation_visualizations.py`: genera visualizaciones cualitativas de segmentacion CT.

## Scripts shell

- `run_xai_explainability.sh`: lanzador para generar Grad-CAM.
- `run_ct_segmentation_visualizations.sh`: lanzador para generar visualizaciones de segmentacion CT.

## Archivo

`archive/` contiene utilidades antiguas o auxiliares que no forman parte del flujo principal de entrega.
