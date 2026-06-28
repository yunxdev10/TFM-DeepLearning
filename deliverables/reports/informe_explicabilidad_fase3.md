# Informe inicial de explicabilidad - Fase 3

Fecha: 2026-05-17

## Objetivo

Iniciar la fase de explicabilidad del TFM mediante mapas Grad-CAM sobre los mejores modelos de clasificacion disponibles y cuantificar la alineacion entre saliencia y mascaras.

## Modelos explicados

| Modalidad | Modelo | Motivo de seleccion |
|---|---|---|
| CXR | `cxr_densenet121_weighted_ce_full` | Mejor resultado CXR consolidado: accuracy `0.9496`, F1-macro `0.9567`, AUC macro `0.9931`. |
| CT | `ct_densenet121_baseline_full` | Mejor F1-macro/AUC en CT entre los candidatos principales, aunque `ct_resnet50_baseline_full` mantiene la mejor accuracy. |
| CT | `ct_resnet50_baseline_full` | Modelo CT con mayor accuracy, usado como comparacion adicional de explicabilidad. |

## Implementacion

Se implemento una base reproducible sin dependencias nuevas:

- `src/evaluation/explainability.py`: Grad-CAM propio en PyTorch, carga de clasificadores, binarizacion de saliencia y metricas de solapamiento.
- `scripts/generate_xai_explanations.py`: generador de figuras y metricas XAI.
- `scripts/run_xai_explainability.sh`: lanzador simple.
- `notebooks/06_explainability.ipynb`: notebook de Fase 3.

La decision final de alcance es usar **Grad-CAM como metodo XAI principal** y no incorporar LIME/SHAP. La razon es metodologica y practica: Grad-CAM ya permite responder la pregunta central de si la evidencia visual del clasificador se concentra en la region disponible, mientras que LIME/SHAP anadirian dependencias, coste computacional y complejidad sin cambiar la conclusion principal observada en CT.

## Metricas generadas

| Modalidad | Modelo | Split mascara | Mascara de referencia | N ejemplos | IoU saliencia-mascara | Ratio saliencia dentro mascara | Pico dentro mascara |
|---|---|---|---|---:|---:|---:|---:|
| CXR | `cxr_densenet121_weighted_ce_full` | test | Mascara pulmonar | 12 | `0.2255` | `0.3143` | `0.3333` |
| CT | `ct_densenet121_baseline_full` | test | Mascara de infeccion | 2 | `0.0146` | `0.0062` | `0.0000` |
| CT | `ct_densenet121_baseline_full` | all | Mascara de infeccion | 4 | `0.0133` | `0.0063` | `0.0000` |
| CT | `ct_resnet50_baseline_full` | all | Mascara de infeccion | 4 | `0.0133` | `0.0083` | `0.0000` |

## Interpretacion

### CXR

La saliencia Grad-CAM tiene una alineacion parcial con el campo pulmonar. El IoU medio `0.2255` y el ratio de saliencia dentro de mascara `0.3143` indican que el modelo no concentra toda la evidencia dentro del pulmon, aunque algunos casos si muestran mapas anatomicos plausibles.

Esta lectura debe formularse con cuidado: las mascaras CXR son mascaras pulmonares, no mascaras de lesion COVID. Por tanto, una alta coincidencia no demostraría localizacion patologica; solo apoyaria que el modelo mira regiones anatomicas razonables.

### CT

En los slices CT anotados del split de test, Grad-CAM no se alinea bien con las pequenas mascaras de infeccion: IoU medio `0.0146`, ratio dentro de lesion `0.0062` y pico dentro de mascara `0.0`.

La ampliacion cualitativa usando todas las mascaras anotadas mantiene la misma conclusion. DenseNet-121 obtiene IoU `0.0133` y ResNet-50 obtiene IoU `0.0133`; en ambos casos el pico de maxima saliencia nunca cae dentro de la mascara de infeccion. El resultado no depende, por tanto, de elegir solo DenseNet como clasificador CT.

Esto es coherente con dos limitaciones observadas antes:

- la clasificacion CT tiene rendimiento menor que CXR;
- las lesiones anotadas pueden ser pequenas, mientras Grad-CAM produce mapas gruesos y de baja resolucion espacial.

El resultado es defendible como hallazgo negativo: el clasificador puede acertar la clase CT-1, pero sus mapas Grad-CAM no demuestran una atencion localizada en la lesion anotada.

## Artefactos

- CXR:
  - `results/explainability/cxr/cxr_densenet121_weighted_ce_full/cxr_densenet121_weighted_ce_full_xai_metrics.csv`
  - `results/explainability/cxr/cxr_densenet121_weighted_ce_full/cxr_densenet121_weighted_ce_full_xai_summary.json`
  - `results/explainability/cxr/cxr_densenet121_weighted_ce_full/figures/`
- CT:
  - `results/explainability/ct/ct_densenet121_baseline_full_test/`
  - `results/explainability/ct/ct_densenet121_baseline_full_all/`
  - `results/explainability/ct/ct_resnet50_baseline_full_all/`

## Siguientes experimentos recomendados

1. Revisar visualmente las figuras CT de DenseNet y ResNet para escoger 2-3 casos representativos para la memoria.
2. Comparar Grad-CAM de prediccion correcta vs incorrecta en CXR para identificar casos "correctos por razones equivocadas".
3. Pasar a `07_final_analysis.ipynb` para consolidar clasificacion, segmentacion y XAI.
