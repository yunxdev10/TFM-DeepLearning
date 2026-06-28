# Informe de calibracion e incertidumbre

Fecha: 2026-05-22

## Objetivo

Anadir un analisis ligero de calibracion probabilistica para los clasificadores CNN ya entrenados, sin reentrenar modelos. La pregunta metodologica es si la confianza asignada por el modelo se corresponde con su probabilidad real de acierto.

## Artefactos creados

- `scripts/build_calibration_analysis.py`
- `notebooks/08_calibration_analysis.ipynb`
- `results/calibration/`

## Salidas generadas

- `results/calibration/calibration_metrics.csv`
- `results/calibration/calibration_bins.csv`
- `results/calibration/high_confidence_errors_top100.csv`
- `results/calibration/selected_models.csv`
- `results/calibration/calibration_summary.md`
- `results/calibration/figures/reliability_diagrams_selected_models.png`
- `results/calibration/figures/confidence_histograms_selected_models.png`
- `results/calibration/figures/ece_by_experiment.png`

## Metricas calculadas

- Confianza maxima media.
- ECE, Expected Calibration Error.
- MCE, Maximum Calibration Error.
- Brier score multiclase.
- Negative log-likelihood.
- Errores de alta confianza, definidos con confianza mayor o igual a `0.90`.

## Resultados iniciales

### Modelos representativos

| Modalidad | Modelo | Accuracy | F1-macro | Confianza media | ECE | Brier |
|---|---|---:|---:|---:|---:|---:|
| CXR | `cxr_densenet121_weighted_ce` | 0.9496 | 0.9567 | 0.9403 | 0.0109 | 0.0801 |
| CT | `ct_resnet50_baseline` | 0.6484 | 0.4043 | 0.6843 | 0.0409 | 0.4870 |
| CT | `ct_densenet121_baseline` | 0.6477 | 0.4173 | 0.6619 | 0.0214 | 0.4844 |

### Lectura

- CXR mantiene muy buen rendimiento y una calibracion razonable en el mejor modelo final.
- CT tiene menor rendimiento y mayor dificultad, pero `ct_densenet121_baseline` presenta mejor ECE que `ct_resnet50_baseline`.
- ResNet-50 CT mantiene la mejor accuracy, pero genera mas errores de alta confianza: `42` errores con confianza `>= 0.90`.
- DenseNet-121 CT tiene F1-macro/AUC ligeramente mejores y menos errores de alta confianza: `14` errores con confianza `>= 0.90`.

## Interpretacion para la memoria

La calibracion complementa las metricas clasicas. En imagen medica, no basta con saber si el modelo acierta; tambien importa si su nivel de confianza es fiable. Un error con alta confianza puede ser especialmente problematico porque el modelo presenta una prediccion incorrecta como segura.

Este analisis refuerza una conclusion metodologica del TFM: CXR no solo obtiene mejor rendimiento, sino tambien un comportamiento probabilistico mas estable. En CT, la menor calidad predictiva debe interpretarse junto con la incertidumbre y los errores de alta confianza, especialmente por tratarse de una tarea de severidad basada en slices 2D.

## Verificacion

- `scripts/build_calibration_analysis.py` compila correctamente.
- `notebooks/08_calibration_analysis.ipynb` es JSON valido.
- Todas las celdas de codigo de `08_calibration_analysis.ipynb` parsean correctamente.
- El script se ejecuto correctamente y genero las tablas/figuras esperadas.
