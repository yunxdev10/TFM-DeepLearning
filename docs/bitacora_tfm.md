# Bitacora de desarrollo del TFM

Este documento recoge, en orden cronologico, las acciones tecnicas realizadas durante el desarrollo del TFM. Su objetivo es servir como base para redactar despues la metodologia, el apartado experimental, las decisiones de diseno y las limitaciones del trabajo.

## Estado general

Proyecto: clasificacion, segmentacion y explicabilidad de COVID-19 en imagen medica usando CXR y CT.

## Principio critico de calidad

Este TFM debe avanzar siempre hacia una solucion real, ideal y defendible academicamente. No se deben usar soluciones dummy, atajos cosmeticos o reducciones de alcance para resolver rapido un bloqueo si eso compromete la validez experimental o metodologica.

Los smoke tests solo sirven para verificar que el pipeline funciona. No deben presentarse como resultados del TFM. Las conclusiones deben basarse en ejecuciones reales, datos completos cuando corresponda, transfer learning correctamente aplicado, evaluacion reproducible y documentacion de la verificacion.

Modalidades consideradas:
- CXR: COVID-19 Radiography Dataset, con 4 clases: COVID, Lung Opacity, Normal y Viral Pneumonia.
- CT: MosMedData, con clases de severidad CT-0, CT-1, CT-2 y CT-3+ tras fusionar CT-3 y CT-4.

Arquitecturas previstas para clasificacion:
- ResNet-50.
- DenseNet-121.
- EfficientNet-B0.

Estrategias previstas para desbalanceo:
- Baseline sin balanceo.
- CrossEntropyLoss con pesos por clase.
- Focal Loss.
- Oversampling mediante WeightedRandomSampler como soporte adicional.

## 2026-05-10 - Fase 0: preprocesamiento y estructura base

### Objetivo

Preparar el proyecto para trabajar de forma modular, reproducible y separada por fases, evitando depender exclusivamente de notebooks.

### Acciones realizadas

- Se organizo el proyecto con modulos bajo `src/`.
- Se centralizaron rutas e hiperparametros en `src/config.py`.
- Se creo el notebook de EDA `notebooks/00_eda.ipynb`.
- Se creo el pipeline de CXR para generar splits estratificados `train/val/test`.
- Se creo el pipeline de CT para transformar volumenes NIfTI de MosMedData a slices 2D PNG.
- Se aplico windowing HU `[-1000, 400]` al preprocesamiento CT, adecuado para tejido pulmonar.
- Se genero metadata CT con `study_id`, `slice_index`, `total_slices`, `image_path` y `label`.
- Se agruparon CT-3 y CT-4 en una unica clase `CT-3+`, porque CT-4 tiene muy pocos estudios para entrenar una clase independiente.
- Se implemento el split CT por `study_id` para evitar fuga de datos entre train, validation y test.

### Decisiones tecnicas

- Para CXR se usan imagenes RGB de tamano 224 x 224, compatibles con pesos ImageNet.
- Para CT se usan imagenes grayscale de tamano 256 x 256.
- En CT se conserva el split por estudio, no por slice, porque dividir slices del mismo paciente entre particiones inflaria artificialmente los resultados.
- CT-3 y CT-4 se fusionan como `CT-3+` para mejorar viabilidad estadistica.

### Verificacion

- CXR produce batches con forma esperada `B x 3 x 224 x 224`.
- CT produce batches con forma esperada `B x 1 x 256 x 256`.
- El preprocesamiento CT genero metadata en `data/MosMedData_Chest_Scan/processed_2d_slices/labels_metadata.csv`.
- El proyecto es importable desde notebooks mediante `src`.

### Estado

Preprocesamiento para clasificacion: completado.

Pendiente para fases posteriores:
- Preparar imagen-mascara para segmentacion CXR.
- Preparar y alinear mascaras CT de lesion/infeccion.
- Reutilizar mascaras para evaluacion cuantitativa de XAI.

## 2026-05-10 - Fase 1: preparacion de clasificacion

### Objetivo

Dejar listo un flujo reproducible para entrenar y evaluar modelos de clasificacion en CXR y CT, primero con smoke tests y despues con experimentos completos.

### Acciones realizadas

- Se implemento `src/models/classifiers.py` con wrapper comun `CovidClassifier`.
- Se soportaron las arquitecturas ResNet-50, DenseNet-121 y EfficientNet-B0.
- Se adapto la primera convolucion para CT con 1 canal.
- Se anadieron aliases de nombres de arquitectura para evitar errores por variantes como `resnet-50` o `efficientnet-b0`.
- Se implementaron transformaciones CXR y CT en `src/data/transforms.py`.
- Se implemento `FocalLoss` y calculo de pesos por clase en `src/training/losses.py`.
- Se anadio soporte de `WeightedRandomSampler` en `src/data/datasets.py`.
- Se creo `src/training/classification_experiment.py` como helper reutilizable para:
  - crear configuraciones de experimento,
  - limitar datasets en modo smoke,
  - construir DataLoaders,
  - seleccionar loss segun estrategia de balanceo,
  - entrenar,
  - evaluar,
  - guardar modelos y artefactos.
- Se regeneraron notebooks de clasificacion:
  - `notebooks/02_cxr_classification.ipynb`,
  - `notebooks/03_ct_classification.ipynb`,
  - `notebooks/04_classification_results.ipynb`.
- Se creo `scripts/run_phase1_smoke.py` para validar CXR y CT desde terminal.

### Decisiones tecnicas

- Se mantiene `RUN_MODE = 'smoke'` por defecto en notebooks para evitar lanzar entrenamientos largos accidentalmente.
- En modo `full`, los notebooks estan preparados para ejecutar 3 arquitecturas x 3 estrategias de balanceo.
- El modo `smoke` usa `pretrained=False`, subset pequeno y 1 epoch. Por tanto, sus metricas no son resultados cientificos; solo validan que el pipeline funciona.
- El modo `full` usa `pretrained=True` y esta preparado para la estrategia de fine-tuning: entrenar cabeza y luego descongelar el modelo completo con learning rate reducido.

### Verificacion ejecutada

Comandos ejecutados con el entorno local `.conda`:

```bash
.conda/bin/python -m compileall src scripts
.conda/bin/python scripts/run_phase1_smoke.py
```

Resultado:
- Compilacion de `src` y `scripts` completada sin errores.
- Smoke CXR completado y artefactos guardados en `results/classification/cxr/`.
- Smoke CT completado y artefactos guardados en `results/classification/ct/`.

Metricas smoke registradas:

| Dataset | Arquitectura | Estrategia | Modo | Accuracy | F1-macro |
|---|---|---|---|---:|---:|
| CXR | ResNet50 | baseline | smoke | 0.25 | 0.10 |
| CT | ResNet50 | baseline | smoke | 0.25 | 0.10 |

Estas metricas son esperables para smoke tests con entrenamiento minimo y no deben interpretarse como rendimiento del modelo.

### Artefactos generados

Para CXR:
- `models/cxr/cxr_resnet50_baseline_smoke.pt`
- `results/classification/cxr/cxr_resnet50_baseline_smoke_summary.json`
- `results/classification/cxr/cxr_resnet50_baseline_smoke_history.csv`
- `results/classification/cxr/cxr_resnet50_baseline_smoke_classification_report.csv`
- `results/classification/cxr/cxr_resnet50_baseline_smoke_confusion_matrix.csv`
- `results/classification/cxr/cxr_resnet50_baseline_smoke_predictions.csv`

Para CT:
- `models/ct/ct_resnet50_baseline_smoke.pt`
- `results/classification/ct/ct_resnet50_baseline_smoke_summary.json`
- `results/classification/ct/ct_resnet50_baseline_smoke_history.csv`
- `results/classification/ct/ct_resnet50_baseline_smoke_classification_report.csv`
- `results/classification/ct/ct_resnet50_baseline_smoke_confusion_matrix.csv`
- `results/classification/ct/ct_resnet50_baseline_smoke_predictions.csv`

### Proximo paso

Ejecutar entrenamientos reales de Fase 1:

1. Abrir `notebooks/02_cxr_classification.ipynb`.
2. Cambiar `RUN_MODE = 'smoke'` por `RUN_MODE = 'full'`.
3. Ejecutar el notebook completo.
4. Repetir el proceso con `notebooks/03_ct_classification.ipynb`.
5. Ejecutar `notebooks/04_classification_results.ipynb` para consolidar resultados.

## Convencion para las siguientes entradas

Cada avance nuevo debe documentarse con esta estructura:

### Objetivo

Que se queria conseguir en ese bloque de trabajo.

### Acciones realizadas

Cambios concretos en codigo, notebooks, datos o resultados.

### Decisiones tecnicas

Motivos de las decisiones importantes y alternativas descartadas si aplica.

### Verificacion

Comandos, notebooks o comprobaciones ejecutadas, con resultado.

### Resultados

Metricas, tablas, artefactos o figuras generadas.

### Pendientes

Lo que queda abierto para la siguiente iteracion.

---

## 2026-05-12 - Fase 1: matriz completa CXR finalizada

### Objetivo

Completar la matriz real de clasificacion CXR en modo `full` para comparar arquitecturas y estrategias de balanceo sobre el COVID-19 Radiography Dataset.

### Acciones realizadas

- Se ejecuto `notebooks/02_cxr_classification.ipynb` en `RUN_MODE = 'full'`.
- Se completaron los 9 experimentos CXR:
  - ResNet-50, DenseNet-121 y EfficientNet-B0.
  - Estrategias `baseline`, `weighted_ce` y `focal_loss`.
- Se guardaron modelos en `models/cxr/`.
- Se guardaron historicos, summaries, classification reports, matrices de confusion y predicciones en `results/classification/cxr/`.

### Decisiones tecnicas

- Se mantuvo la matriz 3 x 3 definida para la Fase 1, sin reducir alcance.
- Tras un reinicio del ordenador durante EfficientNet-B0, se reanudo el notebook saltando experimentos ya persistidos mediante comprobacion de `summary.json` y `.pt`.
- Los resultados registrados corresponden a entrenamiento `full`, no a smoke tests.

### Verificacion

Se comprobaron en disco los 9 summaries `*_full_summary.json` y los 9 modelos `*_full.pt` de CXR:

- `cxr_resnet50_baseline_full`
- `cxr_resnet50_weighted_ce_full`
- `cxr_resnet50_focal_loss_full`
- `cxr_densenet121_baseline_full`
- `cxr_densenet121_weighted_ce_full`
- `cxr_densenet121_focal_loss_full`
- `cxr_efficientnet_b0_baseline_full`
- `cxr_efficientnet_b0_weighted_ce_full`
- `cxr_efficientnet_b0_focal_loss_full`

### Resultados

Mejor resultado CXR por accuracy:

| Experimento | Accuracy | F1-macro | F1-weighted | AUC-ROC macro |
|---|---:|---:|---:|---:|
| `cxr_densenet121_weighted_ce` | 0.9496 | 0.9567 | 0.9496 | 0.9931 |

Resumen interpretativo:

- Las tres familias de modelos superan el umbral objetivo CXR del 90% de accuracy.
- DenseNet-121 con `weighted_ce` es el mejor candidato inicial para CXR.
- `weighted_ce` mejora de forma consistente frente a `baseline` en DenseNet-121 y EfficientNet-B0, y tambien mejora F1-macro en ResNet-50.
- `focal_loss` no supera a `weighted_ce` en esta primera matriz CXR.

### Pendientes

- Ejecutar la matriz `full` equivalente en `notebooks/03_ct_classification.ipynb`.
- Ejecutar `notebooks/04_classification_results.ipynb` cuando esten disponibles CXR y CT completos.
- Revisar matrices de confusion para identificar clases CXR con mayor confusion.

---

## 2026-05-12 - Cribado bibliografico para estado del arte

### Objetivo

Identificar papers cientificos utiles para redactar el estado del arte de la memoria del TFM, valorando su utilidad concreta para clasificacion CXR/CT, segmentacion, explicabilidad y limitaciones metodologicas.

### Acciones realizadas

- Se revisaron referencias sobre:
  - clasificacion COVID-19 en CXR con transfer learning,
  - clasificacion y severidad en CT,
  - datasets COVID-19 Radiography Database y MosMedData,
  - segmentacion con U-Net y Attention U-Net,
  - explicabilidad con Grad-CAM, LIME y SHAP,
  - desbalance y shortcut learning.
- Se creo `docs/estado_arte_papers_tfm.md` con una seleccion priorizada de papers, utilidad para el TFM, puntos redactables y cautelas metodologicas.

### Decisiones tecnicas

- Priorizar papers directamente conectados con las decisiones reales del proyecto: ResNet-50, DenseNet-121, EfficientNet-B0, balanceo de clases, CXR vs CT, segmentacion y XAI.
- Separar fuentes de dataset de papers metodologicos para que la memoria distinga procedencia de datos, arquitectura, explicabilidad y limitaciones.
- Incluir papers criticos sobre shortcut learning para evitar una redaccion excesivamente basada en accuracy.

### Verificacion

- Se comprobaron enlaces y metadatos principales mediante busqueda web.
- El documento generado contiene URLs directas a los articulos o paginas de referencia.

### Resultados

- Nuevo documento de apoyo: `docs/estado_arte_papers_tfm.md`.
- Linea argumental recomendada:
  - CXR y CT son modalidades utiles pero no clinicamente equivalentes.
  - Transfer learning es el enfoque dominante por limitacion de datos anotados.
  - El desbalance exige metricas macro y estrategias como weighted CE/focal loss.
  - La XAI ayuda a auditar atencion del modelo, pero no equivale a segmentacion clinica.
  - La aportacion del TFM es una comparacion controlada y criticamente interpretada, no una arquitectura nueva.

### Pendientes

- Convertir el cribado en texto narrativo para el capitulo de estado del arte.
- Completar CT full para poder conectar literatura y resultados propios.

---

## 2026-05-12 - Matriz comparativa de trabajos relacionados

### Objetivo

Preparar una tabla comparativa, siguiendo la recomendacion del tutor, que muestre que cubren articulos y TFM relacionados y que hueco especifico cubre este TFM.

### Acciones realizadas

- Se buscaron articulos cientificos, revisiones sistematicas y TFM/tesis relacionados con:
  - clasificacion COVID-19 en CXR,
  - clasificacion/severidad COVID-19 en CT,
  - segmentacion pulmonar o de lesion,
  - explicabilidad con Grad-CAM, LIME, SHAP,
  - desbalance de clases,
  - shortcut learning y fuga de datos.
- Se creo `docs/comparativa_trabajos_relacionados.md` con:
  - tabla principal de comparacion,
  - tabla especifica de TFM/tesis localizados,
  - matriz de huecos por dimension,
  - version corta de tabla para adaptar directamente a la memoria,
  - frase de cierre para defender la originalidad del TFM.

### Decisiones tecnicas

- Separar articulos revisados por pares de TFM/tesis o proyectos divulgativos.
- No afirmar una originalidad absoluta, sino una originalidad integradora y metodologica: CXR + CT, clasificacion + segmentacion + XAI, balanceo y control de fuga en CT.
- Incluir trabajos criticos sobre shortcut learning para reforzar la calidad academica de la discusion.

### Verificacion

- Se revisaron fuentes web con enlaces directos a articulos, repositorios institucionales, PubMed/PMC, MDPI, Springer, Scientific Reports, Kaggle y repositorios universitarios.
- Se identifico que algunos TFM aparecen solo como ficha/listado institucional, por lo que se marco cautela para no compararlos en detalle sin memoria completa.

### Resultados

- Nuevo documento: `docs/comparativa_trabajos_relacionados.md`.
- Tesis defendible:
  - la aportacion del TFM no es una nueva arquitectura aislada,
  - sino una comparacion controlada entre modalidades, arquitecturas, balanceo, segmentacion y explicabilidad.

### Pendientes

- Transformar la tabla corta en texto narrativo para el capitulo de estado del arte.
- Anadir resultados CT full cuando esten disponibles para cerrar la comparativa experimental.

---

## 2026-05-13 - Cierre recuperable del notebook 03 CT full

### Objetivo

Terminar y dejar verificable el notebook `notebooks/03_ct_classification.ipynb` tras un reinicio del ordenador durante la ejecucion de la matriz CT en modo `full`.

### Acciones realizadas

- Se comprobaron los artefactos persistidos de CT en `results/classification/ct/` y `models/ct/`.
- Se confirmo que existen los 9 experimentos CT `full` esperados:
  - ResNet-50, DenseNet-121 y EfficientNet-B0.
  - Estrategias `baseline`, `weighted_ce` y `focal_loss`.
- Se actualizo `notebooks/03_ct_classification.ipynb` para trabajar por defecto en `RUN_MODE = 'full'`.
- Se anadio `RESUME_EXISTING = True` para que el notebook reutilice artefactos existentes y no vuelva a entrenar experimentos ya completados.
- Se adapto el experimento de referencia, la matriz completa y las celdas de curvas/matriz de confusion para leer summaries, historicos y matrices ya guardadas cuando el entrenamiento ya existe.

### Decisiones tecnicas

- No repetir entrenamientos completos si ya existen simultaneamente el modelo `.pt` y el `summary.json`, porque esos dos artefactos prueban que el experimento se guardo correctamente.
- Mantener la matriz experimental completa 3 x 3 en CT, sin reducir alcance tras el reinicio.
- Hacer el notebook reanudable para minimizar el riesgo de perder progreso en futuros cortes o reinicios.

### Verificacion

- Se valido que `notebooks/03_ct_classification.ipynb` sigue siendo JSON valido.
- Se compilaron sintacticamente todas las celdas de codigo del notebook.
- Se comprobaron en disco los 9 modelos `models/ct/*_full.pt`.
- Se comprobaron los 9 summaries `results/classification/ct/*_full_summary.json`, sin archivos faltantes, vacios o JSON corruptos.

### Resultados

Mejores resultados CT por accuracy:

| Experimento | Accuracy | F1-macro | F1-weighted | AUC-ROC macro |
|---|---:|---:|---:|---:|
| `ct_resnet50_baseline_full` | 0.6484 | 0.4043 | 0.6056 | 0.7200 |
| `ct_densenet121_baseline_full` | 0.6477 | 0.4173 | 0.6061 | 0.7229 |
| `ct_efficientnet_b0_baseline_full` | 0.6195 | 0.3897 | 0.5757 | 0.6943 |

Resumen interpretativo inicial:

- CT queda completado en modo `full` para la Fase 1.
- Las variantes `baseline` obtienen mayor accuracy que `weighted_ce` y `focal_loss` en esta primera matriz CT.
- DenseNet-121 baseline ofrece el mejor F1-macro y AUC-ROC macro, aunque ResNet-50 baseline queda ligeramente por encima en accuracy.
- Los resultados CT son claramente inferiores a CXR, por lo que sera importante discutir dificultad del dataset, severidad, ruido de slices 2D y desbalance.

### Pendientes

- Ejecutar visualmente `notebooks/03_ct_classification.ipynb` desde Jupyter para refrescar las salidas del notebook; con `RESUME_EXISTING = True` deberia saltar entrenamientos ya guardados.
- Ejecutar `notebooks/04_classification_results.ipynb` para consolidar CXR + CT.
- Revisar matrices de confusion CT para interpretar errores por clase de severidad.

---

## 2026-05-14 - Informe interpretativo del notebook 04

### Objetivo

Interpretar de forma detallada las tablas, graficas, matrices de confusion, curvas ROC y lectura cross-modal del notebook `notebooks/04_classification_results.ipynb`.

### Acciones realizadas

- Se reconstruyeron las tablas de resultados desde los artefactos `full` guardados en `results/classification/`.
- Se interpretaron los resultados CXR y CT por arquitectura, estrategia de balanceo y metrica.
- Se analizaron las matrices de confusion de los modelos clave:
  - `cxr_densenet121_weighted_ce_full`,
  - `ct_resnet50_baseline_full`,
  - `ct_densenet121_baseline_full`.
- Se calcularon e interpretaron AUC one-vs-rest por clase para los modelos principales.
- Se creo el informe `docs/informe_resultados_clasificacion_fase1.md`.

### Decisiones tecnicas

- Para la interpretacion cientifica se usaron solo experimentos `full`; los `smoke` quedan como verificacion de pipeline.
- Se priorizo F1-macro para elegir mejores modelos por dataset, porque el desbalance puede hacer enganosa la accuracy.
- En CT se distinguio entre mejor modelo por accuracy (`ct_resnet50_baseline_full`) y mejor modelo por equilibrio macro/AUC (`ct_densenet121_baseline_full`).

### Verificacion

- Se comprobaron los 18 summaries `full` esperados: 9 CXR y 9 CT.
- Se leyeron classification reports, matrices de confusion y predicciones guardadas.
- Se verifico que CXR supera el objetivo de accuracy >= 90%.
- Se verifico que CT no alcanza el objetivo inicial de accuracy >= 85%, por lo que debe discutirse como limitacion experimental.

### Resultados

- Mejor CXR: `cxr_densenet121_weighted_ce_full`, con accuracy 0.9496, F1-macro 0.9567 y AUC macro 0.9931.
- Mejor CT por accuracy: `ct_resnet50_baseline_full`, con accuracy 0.6484.
- Mejor CT por F1-macro/AUC: `ct_densenet121_baseline_full`, con F1-macro 0.4173 y AUC macro 0.7229.
- CXR presenta errores principalmente entre `Lung_Opacity` y `Normal`.
- CT presenta una tendencia clara a predecir `CT-1`, con bajo recall en `CT-2` y `CT-3+`.

### Pendientes

- Refrescar visualmente las salidas del notebook `04` si se quiere conservar el notebook ejecutado.
- Iniciar Fase 2: segmentacion pulmonar en CXR y segmentacion de lesion/infeccion en CT.

---

## 2026-05-14 - Inicio Fase 2: pipeline entregable de segmentacion

### Objetivo

Iniciar la Fase 2 con un pipeline real y reproducible para segmentacion pulmonar en CXR y segmentacion de infeccion/lesion en CT, evitando placeholders y dejando hiperparametros preparados para experimentacion.

### Acciones realizadas

- Se auditaron las mascaras disponibles:
  - CXR: 21.165 pares imagen-mascara distribuidos por las cuatro clases del COVID-19 Radiography Dataset.
  - CT: 50 estudios MosMedData con mascaras `.nii` de infeccion (`study_0255` a `study_0304`).
- Se genero el dataset CT de segmentacion en slices 2D pareados imagen-mascara, conservando solo slices con mascara positiva.
- Se implemento `src/data/segmentation.py` con:
  - construccion de dataframe CXR imagen-mascara,
  - construccion de dataframe CT desde volumenes `.nii` y mascaras `.nii`,
  - split reproducible con separacion por `study_id` en CT,
  - dataset PyTorch pareado imagen-mascara,
  - transformaciones sincronizadas para imagen y mascara.
- Se implemento `src/models/segmentation.py` con:
  - U-Net,
  - Attention U-Net,
  - constructor `build_segmentation_model`.
- Se implemento `src/training/segmentation_experiment.py` con:
  - Dice + BCE loss,
  - metricas Dice, IoU y pixel accuracy,
  - trainer con early stopping por Dice de validacion,
  - guardado de modelo, historico y summary JSON,
  - configuracion `smoke` y `full`.
- Se creo `notebooks/05_segmentation.ipynb` como notebook de ejecucion de Fase 2.

### Decisiones tecnicas

- No se anadio `segmentation_models_pytorch` porque el proyecto mantiene la regla de no introducir dependencias nuevas sin necesidad explicita; se implementaron U-Net y Attention U-Net directamente en PyTorch.
- CXR se trabaja como segmentacion binaria de campo pulmonar usando las mascaras disponibles del dataset.
- CT se trabaja como segmentacion binaria de infeccion/lesion usando solo los 50 estudios anotados.
- En CT, los splits se hacen por `study_id` para evitar fuga entre slices del mismo estudio.
- El notebook deja `HYPERPARAMETER_OVERRIDES` para ajustar batch size, learning rate, base features y epochs durante la experimentacion real.

### Verificacion

- Se valido la sintaxis de:
  - `src/data/segmentation.py`,
  - `src/models/segmentation.py`,
  - `src/training/segmentation_experiment.py`,
  - `notebooks/05_segmentation.ipynb`.
- Se comprobo forward pass de U-Net y Attention U-Net con salida `(batch, 1, H, W)`.
- Se reconstruyeron los pares disponibles:
  - CXR: 21.165 pares.
  - CT: 704 slices con mascara positiva procedentes de 50 estudios.
- Se ejecuto una prueba smoke de entrenamiento U-Net CXR con 64 imagenes de train, 24 de validacion y 24 de test, verificando calculo de loss, Dice, IoU y pixel accuracy.

### Resultados

- La Fase 2 ya tiene una base de codigo ejecutable y entregable.
- Los datos reales disponibles confirman que CXR tiene volumen suficiente para resultados robustos.
- CT tiene anotacion limitada pero util para evaluar segmentacion de lesion; debe interpretarse con cautela por el bajo numero de estudios anotados.
- Smoke CXR U-Net completado y guardado en `results/segmentation/cxr/` y `models/segmentation/cxr/`. Sus metricas no son resultados cientificos, solo validacion del pipeline.

### Pendientes

- Ejecutar `notebooks/05_segmentation.ipynb` primero en `RUN_MODE = 'smoke'`.
- Cambiar a `RUN_MODE = 'full'` para entrenar U-Net y Attention U-Net en CXR y CT.
- Comparar Dice, IoU y pixel accuracy por dataset y arquitectura.
- Revisar visualmente predicciones de segmentacion antes de aceptar resultados para la memoria.

---

## 2026-05-14 - Diagnostico y ajuste de hiperparametros para segmentacion CT

### Objetivo

Analizar los resultados full de segmentacion y preparar una estrategia de mejora para CT, manteniendo criterios de resultado real y defendible.

### Acciones realizadas

- Se revisaron los summaries e historicos full de segmentacion.
- Se calculo la fraccion de pixeles positivos en las mascaras CT:
  - train: 0.005717,
  - val: 0.003620,
  - test: 0.003499.
- Se realizo un barrido de umbral sobre los modelos CT ya entrenados.
- Se amplio `src/training/segmentation_experiment.py` para soportar:
  - variantes con nombre propio,
  - `weighted_dice_bce`,
  - `weighted_tversky_bce`,
  - `pos_weight`,
  - optimizacion de umbral en validacion.
- Se actualizo `notebooks/05_segmentation.ipynb` con un bloque de hiperparametros recomendado para CT.
- Se creo `docs/informe_ajuste_segmentacion_ct.md`.

### Decisiones tecnicas

- No se toma `pixel_accuracy` como metrica principal en CT porque el fondo domina mas del 99% de los pixeles.
- Se mantienen Dice e IoU como metricas centrales.
- Para CT se propone probar Tversky+BCE ponderada, porque permite penalizar mas los falsos negativos de lesion.
- Las variantes de ajuste se guardaran con `variant_name` para no sobrescribir los resultados baseline.

### Verificacion

- Se comprobo que el mejor CXR es estable: Dice aproximado 0.985.
- Se comprobo que CT queda alrededor de Dice 0.48-0.50.
- El barrido de umbral no aporta una mejora sustancial:
  - U-Net CT mejora solo de 0.4790 a 0.4933 con umbral 0.4.
  - Attention U-Net CT mantiene su mejor resultado alrededor de umbral 0.5.
- Se verifico sintaxis tras modificar el codigo de entrenamiento.

### Resultados

- CXR no necesita ajuste urgente: los resultados son muy fuertes y defendibles.
- CT no esta fallando por un simple umbral; el problema principal es el desbalance extremo y la baja cantidad de estudios anotados.
- Queda preparado el pipeline para experimentar CT con perdidas ponderadas sin contaminar los resultados baseline.

### Pendientes

- Ejecutar una variante CT de bajo coste con `weighted_tversky_bce`, `pos_weight` entre 20 y 50 y `optimize_threshold=True`.
- Si mejora Dice de validacion, repetir la mejor variante con mas capacidad o mas epochs.
- Revisar visualmente predicciones CT para distinguir infrasegmentacion de sobresegmentacion.

---

## 2026-05-14 - Resultado variante CT Tversky ponderada

### Objetivo

Evaluar si una variante ligera de Attention U-Net con Tversky+BCE ponderada mejora la segmentacion CT de lesion frente al baseline.

### Acciones realizadas

- Se ejecuto `notebooks/05b_ct_tversky_variant.ipynb`.
- Se entreno la variante `ct_attention_unet_tversky_pos30_bf16_thr_segmentation_full`.
- Configuracion principal:
  - `loss_name = weighted_tversky_bce`,
  - `pos_weight = 30.0`,
  - `base_features = 16`,
  - `epochs = 12`,
  - `batch_size = 8`,
  - `optimize_threshold = True`.

### Resultados

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_tversky_pos30_bf16_thr_segmentation_full` | 0.5002 | 0.3633 | 0.9963 | 0.8 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.5 |
| `ct_unet_segmentation_full` | 0.4790 | 0.3522 | 0.9971 | 0.5 |

### Interpretacion

- La variante Tversky ponderada no aporta una mejora material frente al baseline Attention U-Net.
- El Dice sube solo de 0.5001 a 0.5002, una diferencia despreciable.
- El IoU baja ligeramente de 0.3648 a 0.3633.
- El umbral optimizado sube a 0.8, lo que sugiere que la variante tiende a necesitar una binarizacion mas estricta para controlar falsos positivos.
- Dado que la mejora no es clara, no conviene presentar esta variante como superior.

### Decision tecnica

Mantener `ct_attention_unet_segmentation_full` como referencia CT principal de segmentacion, salvo que una revision visual demuestre una ventaja cualitativa clara de la variante Tversky. La diferencia numerica no justifica repetir una version pesada solo por hiperparametros.

### Pendientes

- Revisar visualmente ejemplos CT del baseline y de la variante para detectar si una segmenta lesiones de forma mas coherente.
- Si se busca mejora adicional, priorizar cambios de datos o muestreo antes que seguir ajustando solo la loss.

---

## 2026-05-14 - Preparacion de experimento CT con patch training

### Objetivo

Preparar un experimento CT que modifique la estrategia de datos, no solo hiperparametros, para afrontar el desbalance extremo de pixeles de lesion.

### Acciones realizadas

- Se amplio `SegmentationPairTransform` para admitir:
  - `train_crop_size`,
  - `positive_crop_prob`.
- Se amplio `SegmentationRunConfig` para guardar esos parametros como parte del experimento.
- Se creo `notebooks/05c_ct_patch_training.ipynb`.
- El nuevo experimento entrena con parches 128x128 sesgados hacia pixeles positivos de lesion, pero valida y testea sobre slices completos 256x256.

### Decisiones tecnicas

- Usar la mascara para seleccionar parches solo durante entrenamiento es aceptable como estrategia supervisada de muestreo.
- No se usa la mascara para recortar validacion ni test, para evitar fuga de informacion y resultados artificialmente inflados.
- La variante se guarda como `ct_attention_unet_patch128_pos80_tversky_pos20_thr_segmentation_full`, sin pisar baselines.

### Verificacion

- Se valido la sintaxis de `src/data/segmentation.py`, `src/training/segmentation_experiment.py` y `notebooks/05c_ct_patch_training.ipynb`.
- Se comprobo que el transform de entrenamiento devuelve parches `(1, 128, 128)`.
- Se comprobo que el transform de evaluacion mantiene slices completos `(1, 256, 256)`.

### Pendientes

- Ejecutar `notebooks/05c_ct_patch_training.ipynb`.
- Comparar Dice/IoU contra `ct_attention_unet_segmentation_full` y la variante Tversky previa.
- Revisar visualmente si el patch training reduce infrasegmentacion sin disparar falsos positivos.

---

## 2026-05-14 - Resultado CT patch training

### Objetivo

Evaluar si entrenar CT con parches 128x128 sesgados hacia lesion mejora la segmentacion de infeccion frente a entrenar con slices completos.

### Acciones realizadas

- Se ejecuto `notebooks/05c_ct_patch_training.ipynb`.
- Se entreno `ct_attention_unet_patch128_pos80_tversky_pos20_thr_segmentation_full`.
- La evaluacion se mantuvo sobre slices completos 256x256 para evitar fuga de informacion.

### Resultados

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_tversky_pos30_bf16_thr_segmentation_full` | 0.5002 | 0.3633 | 0.9963 | 0.8 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.5 |
| `ct_unet_segmentation_full` | 0.4790 | 0.3522 | 0.9971 | 0.5 |
| `ct_attention_unet_patch128_pos80_tversky_pos20_thr_segmentation_full` | 0.4745 | 0.3383 | 0.9944 | 0.8 |

### Interpretacion

- El patch training no mejora CT; empeora Dice e IoU frente al baseline Attention U-Net.
- La bajada de Dice de 0.5001 a 0.4745 indica que aprender en parches positivos no se transfiere bien a la segmentacion de slices completos.
- La perdida de contexto anatomico global probablemente perjudica al modelo: al entrenar en parches, ve mejor la lesion local, pero pierde informacion de posicion y estructura pulmonar completa.
- El umbral optimizado vuelve a ser 0.8, coherente con una tendencia a sobreactivar lesion que necesita binarizacion estricta.

### Decision tecnica

Descartar patch training como mejora principal para CT. Mantener `ct_attention_unet_segmentation_full` como referencia principal por equilibrio Dice/IoU. La variante Tversky ponderada puede reportarse como experimento adicional, pero no como mejora material.

### Pendientes

- Revisar visualizaciones cualitativas de CT para entender si el error principal es infrasegmentacion, sobresegmentacion o mala localizacion.
- Para intentar una mejora adicional, priorizar enfoques que conserven contexto global: por ejemplo, entrada completa con ponderacion por area/lesion o postprocesado de componentes pequenas, antes que parches aislados.

---

## 2026-05-14 - Preparacion de mejoras CT: postprocesado y mixed context

### Objetivo

Preparar nuevas vias de mejora CT tras comprobar que Tversky pura y patch training no mejoran materialmente el baseline.

### Acciones realizadas

- Se evaluo postprocesado sobre modelos CT existentes.
- Se implemento `src/evaluation/segmentation_postprocessing.py` con:
  - filtrado por componentes conectadas,
  - cierre morfologico opcional,
  - calculo de Dice, IoU y pixel accuracy para mascaras binarias.
- Se amplio el transform de segmentacion con `train_crop_prob` para permitir entrenamiento mixto:
  - parte de batches con slice completo reescalado,
  - parte con patch centrado en lesion.
- Se creo `notebooks/05d_ct_mixed_context_training.ipynb`.
- Se creo `notebooks/05e_ct_postprocessing_analysis.ipynb`.

### Resultados preliminares

- El postprocesado mejora solo marginalmente el baseline Attention U-Net CT:
  - Dice baseline: 0.5001.
  - Mejor ajuste postprocesado probado: Dice 0.5042 con threshold 0.45 y area minima 5 pixeles.
- La mejora es pequena y no debe venderse como salto relevante, pero puede reportarse como ajuste de postprocesado.

### Decision tecnica

Probar `05d_ct_mixed_context_training.ipynb` como siguiente experimento estructural, porque combina contexto global y foco local sin usar la mascara para recortar validacion/test.

### Verificacion

- Se valido que el entrenamiento mixto produce tensores `(1, 160, 160)` tanto para slices completos reescalados como para patches.
- Se valido que evaluacion/test mantienen slices completos `(1, 256, 256)`.
- Se valido sintaxis de los nuevos notebooks.

### Pendientes

- Ejecutar `notebooks/05d_ct_mixed_context_training.ipynb`.
- Ejecutar `notebooks/05e_ct_postprocessing_analysis.ipynb` si se desea reportar postprocesado.
- Comparar resultados contra `ct_attention_unet_segmentation_full` y decidir si alguna mejora es material.

---

## 2026-05-15 - Resultado CT mixed context training

### Objetivo

Evaluar si una estrategia mixta de entrenamiento CT, combinando contexto global y parches centrados en lesion, mejora la segmentacion de infeccion frente a los baselines.

### Acciones realizadas

- Se ejecuto `notebooks/05d_ct_mixed_context_training.ipynb`.
- Se entreno `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full`.
- La estrategia combina:
  - 50% slices completos reescalados durante entrenamiento,
  - 50% patches 160x160,
  - 80% de probabilidad de centrar el patch en pixeles positivos de lesion,
  - Tversky+BCE ponderada con `pos_weight = 20`.
- Validacion y test se mantienen sobre slices completos 256x256.

### Resultados

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.8 |
| `ct_attention_unet_tversky_pos30_bf16_thr_segmentation_full` | 0.5002 | 0.3633 | 0.9963 | 0.8 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.5 |
| `ct_unet_segmentation_full` | 0.4790 | 0.3522 | 0.9971 | 0.5 |
| `ct_attention_unet_patch128_pos80_tversky_pos20_thr_segmentation_full` | 0.4745 | 0.3383 | 0.9944 | 0.8 |

### Interpretacion

- Mixed context training mejora de forma material el baseline CT.
- Frente a `ct_attention_unet_segmentation_full`, Dice sube de 0.5001 a 0.5242 y IoU de 0.3648 a 0.3846.
- La mejora absoluta es aproximadamente:
  - +0.0241 en Dice,
  - +0.0198 en IoU.
- Patch training puro empeoraba, por lo que el resultado sugiere que el contexto anatomico global sigue siendo importante.
- La combinacion de contexto global + parches positivos parece ser mejor compromiso para lesiones pequenas y desbalanceadas.
- El threshold optimizado vuelve a ser 0.8, coherente con las variantes Tversky.

### Decision tecnica

Adoptar `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` como mejor modelo CT de segmentacion hasta nueva evidencia. Reportar el baseline, Tversky, patch puro y mixed context como secuencia experimental defendible.

### Pendientes

- Revisar visualmente ejemplos de prediccion del modelo mixed context.
- Evaluar si aplicar postprocesado ligero al modelo mixed context aporta mejora adicional.
- Actualizar el informe de Fase 2 con comparativa CXR/CT y explicacion metodologica.

---

## 2026-05-15 - Analisis adicional CT y preparacion de variante 2.5D

### Objetivo

Analizar por que la segmentacion CT sigue limitada tras mixed context training y preparar una variante experimental con mayor contexto anatomico.

### Acciones realizadas

- Se audito el modelo `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full`.
- Se realizo un barrido diagnostico de threshold en validacion y test para baseline y mixed context.
- Se evaluo postprocesado ligero sobre el modelo mixed context.
- Se analizo el rendimiento por tamano de lesion y por estudio.
- Se implemento soporte para CT 2.5D en `SegmentationDataset`, cargando `z-1`, `z` y `z+1` como tres canales cuando `ct_context_slices=True`.
- Se amplio el barrido automatico de threshold hasta `0.95`.
- Se creo `notebooks/05f_ct_25d_context_training.ipynb`.
- Se creo `docs/informe_mejora_segmentacion_ct.md`.

### Resultados del analisis

- El modelo mixed context guardado obtiene Dice `0.5242` e IoU `0.3846` con threshold `0.8`.
- En barrido diagnostico sobre test, mixed context alcanza Dice `0.5387` e IoU `0.4003` con threshold `0.90`; este valor se interpreta como diagnostico, no como resultado final principal seleccionado de forma independiente.
- El postprocesado morfologico sobre mixed context aporta una mejora minima: Dice `0.5389`, IoU `0.4007`.
- TTA horizontal no mejora el promedio: Dice `0.5297`.
- El error se concentra en lesiones pequenas:
  - `<=50 px`: baseline Dice `0.3072`, mixed Dice `0.3720`.
  - `51-150 px`: baseline Dice `0.4522`, mixed Dice `0.5063`.
  - `601-1200 px`: baseline Dice `0.7036`, mixed Dice `0.7223`.
- Los estudios mas dificiles del test actual son `study_0285`, `study_0268` y `study_0294`.

### Decision tecnica

Priorizar la variante CT 2.5D como siguiente experimento, porque CT es volumetrico y el principal margen de mejora esta en lesiones pequenas/ambiguas donde los slices vecinos pueden aportar contexto.

### Verificacion

- `src/data/segmentation.py` y `src/training/segmentation_experiment.py` compilan correctamente.
- El notebook `notebooks/05f_ct_25d_context_training.ipynb` es JSON valido.
- Se verifico que `SegmentationDataset(..., in_channels=3, ct_context_slices=True)` entrega tensores `(3, 256, 256)` y mascaras `(1, 256, 256)`.
- Se ejecuto un smoke training 2.5D de 1 epoca con pocas muestras para comprobar entrenamiento, threshold tuning y evaluacion.

### Pendientes

- Ejecutar `notebooks/05f_ct_25d_context_training.ipynb` completo.
- Comparar el resultado 2.5D contra mixed context 2D.
- Si mejora, ejecutar variantes con patch `192x192` y menor `pos_weight`.
- Si no mejora, reportarlo como experimento negativo y mantener mixed context 2D como mejor modelo CT.

---

## 2026-05-15 - Resultado CT 2.5D y siguiente ablacion mixed-context

### Objetivo

Interpretar el resultado de `05f_ct_25d_context_training.ipynb` y decidir el siguiente ajuste experimental para CT.

### Resultado

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.80 |
| `ct_attention_unet_ct25d_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5143 | 0.3843 | 0.9972 | 0.95 |

### Interpretacion

- CT 2.5D no supera al mejor mixed-context 2D.
- IoU queda practicamente empatado, pero Dice baja de `0.5242` a `0.5143`.
- El threshold `0.95` y la pixel accuracy mas alta indican que CT 2.5D es mas conservador y predice menos lesion.
- En test, CT 2.5D predice de media `249.7` pixeles positivos frente a `399.8` del mixed-context 2D, con objetivo real medio `229.3`.
- Por tamano de lesion:
  - `<=50 px`: mixed 2D Dice `0.3471`, CT 2.5D Dice `0.2851`.
  - `151-300 px`: mixed 2D Dice `0.5439`, CT 2.5D Dice `0.5491`.
  - `301-600 px`: mixed 2D Dice `0.6543`, CT 2.5D Dice `0.6576`.
- CT 2.5D ayuda ligeramente en lesiones medias, pero empeora lesiones muy pequenas, que son el mayor cuello de botella.

### Decision tecnica

Mantener `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` como mejor modelo CT. Reportar CT 2.5D como experimento negativo/neutral.

Preparar una nueva ablacion 2D mas conservadora:

- patch `192x192`,
- `train_crop_prob = 0.3`,
- `positive_crop_prob = 0.7`,
- `pos_weight = 10`,
- threshold tuning limitado a `0.80` para comparabilidad con el mejor mixed-context 2D actual.

### Cambios realizados

- Se creo `notebooks/05g_ct_mixed_context_ablation.ipynb`.
- Se hizo configurable el rango de busqueda de threshold en `src/training/segmentation_experiment.py`.
- Se actualizo `docs/informe_mejora_segmentacion_ct.md`.

### Verificacion

- `src/training/segmentation_experiment.py` y `src/data/segmentation.py` compilan correctamente.
- `notebooks/05g_ct_mixed_context_ablation.ipynb` es JSON valido.

### Pendientes

- Ejecutar `notebooks/05g_ct_mixed_context_ablation.ipynb`.
- Comparar la nueva variante contra Dice `0.5242` e IoU `0.3846`.

---

## 2026-05-15 - Nuevo mejor modelo CT individual y preparacion de ensemble

### Objetivo

Interpretar el resultado de `05g_ct_mixed_context_ablation.ipynb` y decidir si conviene seguir entrenando o combinar modelos.

### Resultado

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.80 |
| `ct_attention_unet_ct25d_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5143 | 0.3843 | 0.9972 | 0.95 |

### Interpretacion

- La nueva ablacion 2D es el mejor modelo CT individual hasta ahora.
- Frente al mixed-context anterior:
  - Dice mejora `+0.0062`.
  - IoU mejora `+0.0096`.
  - La prediccion media baja de `399.8` a `342.3` pixeles positivos, mas cerca del objetivo real medio de `229.3`.
- Por tamano de lesion:
  - `<=50 px`: Dice sube de `0.3471` a `0.3637`.
  - `51-150 px`: Dice baja de `0.4792` a `0.4505`.
  - `151-300 px`: Dice sube de `0.5439` a `0.5538`.
  - `301-600 px`: Dice sube de `0.6543` a `0.6995`.
  - `601-1200 px`: Dice sube de `0.7036` a `0.7240`.
- La mejora viene principalmente de conservar mas contexto y reducir sobresegmentacion, aunque hay perdida en lesiones pequenas/medianas.

### Decision tecnica

Adoptar `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` como mejor modelo CT individual.

Como los dos mejores modelos son complementarios por estudio y tamano de lesion, preparar una evaluacion de ensemble en vez de lanzar inmediatamente otra variante de entrenamiento.

### Cambios realizados

- Se creo `notebooks/05h_ct_mixed_ensemble.ipynb`.
- Se actualizo `docs/informe_mejora_segmentacion_ct.md`.

### Verificacion

- Se compararon umbrales diagnosticos de la nueva variante en validacion y test.
- Se comparo el rendimiento por tamano de lesion y por estudio frente al mixed-context anterior.
- El notebook de ensemble queda preparado para seleccionar peso y threshold en validacion y evaluar test una sola vez.

### Pendientes

- Ejecutar `notebooks/05h_ct_mixed_ensemble.ipynb`.
- Si el ensemble mejora, reportarlo como resultado complementario al mejor modelo individual.
- Si el ensemble no mejora, mantener `mixed30_patch192_pos70_tversky_pos10_thr080` como resultado CT final.

---

## 2026-05-15 - Resultado ensemble CT validado

### Objetivo

Evaluar si la combinacion de los dos mejores modelos mixed-context mejora el resultado CT sin entrenar una arquitectura nueva.

### Protocolo

- Se ejecuto `notebooks/05h_ct_mixed_ensemble.ipynb`.
- Se combinaron probabilidades de:
  - `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full`.
  - `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full`.
- El peso del ensemble y el threshold se seleccionaron en validacion.
- Test se evaluo una vez con la configuracion elegida en validacion.

### Seleccion en validacion

| old_weight | new_weight | Threshold | Val Dice | Val IoU | Val pixel accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.20 | 0.90 | 0.5153 | 0.3906 | 0.9969 |

### Resultado en test

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_ensemble_mixed50_mixed30_segmentation_full` | 0.5444 | 0.4097 | 0.9972 | 0.90 |
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.80 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.50 |

### Interpretacion

- El ensemble es el mejor resultado CT global hasta ahora.
- Mejora frente al mejor modelo individual:
  - Dice `+0.0140`.
  - IoU `+0.0156`.
- Mejora frente al baseline Attention U-Net:
  - Dice `+0.0443`.
  - IoU `+0.0449`.
- La eleccion `old_weight=0.80`, `new_weight=0.20` indica que el modelo mixed-context original sigue aportando sensibilidad, mientras que la variante `patch192/pos10` corrige parte de la sobresegmentacion.
- El resultado es defendible porque los hiperparametros del ensemble se eligieron en validacion, no en test.
- Debe indicarse la limitacion de tamano muestral CT: solo 50 estudios anotados y 8 estudios en test.

### Decision tecnica

Reportar:

- `ct_attention_unet_ensemble_mixed50_mixed30_segmentation_full` como mejor resultado CT global.
- `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` como mejor modelo CT individual.

No seguir entrenando variantes CT por ahora. El siguiente paso debe ser cierre de Fase 2: visualizaciones cualitativas, tabla comparativa final e informe de resultados.

### Verificacion

- Se confirmo que el summary del ensemble quedo guardado en `results/segmentation/ct`.
- Se compararon las mejoras absolutas frente al mejor modelo individual y frente al baseline Attention U-Net.

### Pendientes

- Generar visualizaciones cualitativas representativas del ensemble/mejor modelo individual.
- Actualizar o crear informe final de segmentacion Fase 2.
- Preparar la transicion a la siguiente fase del TFM.

---

## 2026-05-15 - Preparacion de experimentos adicionales CT

### Objetivo

Responder si aun existen vias razonables para mejorar CT tras obtener el ensemble validado con Dice `0.5444` e IoU `0.4097`.

### Analisis

- Entrenar mas tiempo sin cambiar nada no parece la mejor apuesta:
  - los historiales ya muestran oscilacion de validacion;
  - el entrenamiento usa early stopping;
  - el dataset CT anotado es pequeno.
- Las vias con mas sentido experimental son:
  - modificar datos para reducir falsos positivos;
  - aumentar capacidad de forma controlada y con early stopping.

### Cambios realizados

- Se creo `notebooks/05i_ct_negative_slice_training.ipynb`.
- Se creo `notebooks/05j_ct_high_capacity_refinement.ipynb`.
- Se actualizo `docs/informe_mejora_segmentacion_ct.md` con la justificacion de ambos experimentos.

### Experimento 05i

Entrenamiento con slices negativos:

- Usa los mismos estudios de train/val/test.
- Amplia solo train con slices negativos de estudios de train.
- Mantiene val/test positivos para comparabilidad.
- Ratio inicial: `1` negativo por cada positivo.
- Objetivo: reducir sobresegmentacion y falsos positivos.

### Experimento 05j

Refinamiento de mayor capacidad:

- Replica la mejor receta individual `mixed30_patch192_pos70_tversky_pos10`.
- Aumenta `base_features` de `16` a `32`.
- Usa `batch_size = 4`, `epochs = 36`, `early_stopping_patience = 8`.
- Objetivo: comprobar si la arquitectura estaba limitada por capacidad.

### Decision tecnica

Ejecutar primero `05i`, porque es una mejora de datos y puede aportar interpretacion metodologica. Ejecutar `05j` despues solo si se quiere invertir mas tiempo de computo.

Comparar cualquier nuevo resultado contra:

- Ensemble CT global: Dice `0.5444`, IoU `0.4097`.
- Mejor CT individual: Dice `0.5304`, IoU `0.3942`.

### Verificacion

- `notebooks/05i_ct_negative_slice_training.ipynb` es JSON valido.
- `notebooks/05j_ct_high_capacity_refinement.ipynb` es JSON valido.
- `src/data/segmentation.py` y `src/training/segmentation_experiment.py` compilan correctamente.

### Pendientes

- Ejecutar `05i`.
- Si `05i` empeora por exceso de conservadurismo, repetirlo con `NEGATIVE_RATIO = 0.5`.
- Ejecutar `05j` si se desea probar una variante mas lenta y de mayor capacidad.

---

## 2026-05-15 - Resultado CT high-capacity refinement

### Objetivo

Interpretar el resultado de `05j_ct_high_capacity_refinement.ipynb`, que prueba si aumentar capacidad mejora la segmentacion CT.

### Resultado

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full` | 0.5637 | 0.4305 | 0.9977 | 0.90 |
| `ct_attention_unet_ensemble_mixed50_mixed30_segmentation_full` | 0.5444 | 0.4097 | 0.9972 | 0.90 |
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.50 |

### Interpretacion

- La variante bf32 es el nuevo mejor resultado CT global e individual.
- Frente al ensemble anterior:
  - Dice mejora `+0.0193`.
  - IoU mejora `+0.0208`.
- Frente al mejor modelo individual anterior:
  - Dice mejora `+0.0333`.
  - IoU mejora `+0.0364`.
- Frente al baseline Attention U-Net:
  - Dice mejora `+0.0636`.
  - IoU mejora `+0.0657`.
- La mejora indica que la receta `mixed30_patch192_pos70_tversky_pos10` estaba limitada por capacidad con `base_features=16`.
- El aumento a `base_features=32` fue beneficioso al combinarse con `batch_size=4`, mas epocas y early stopping.

### Analisis adicional

- Se probo un ensemble diagnostico entre bf32, el mixed-context original y la variante bf16.
- La seleccion en validacion eligio `100%` bf32 con threshold `0.90`.
- Por tanto, no merece crear un ensemble nuevo con bf32: el modelo bf32 puro domina a los anteriores en validacion.
- En diagnostico test, thresholds mas bajos pueden subir mas el Dice, pero no deben usarse como resultado principal porque el threshold oficial debe venir de validacion.

### Decision tecnica

Adoptar `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full` como mejor resultado CT final por ahora.

Mantener:

- ensemble anterior como experimento intermedio;
- `mixed30_patch192_pos70_tversky_pos10_thr080` como mejor modelo ligero;
- bf32 como mejor modelo individual/global.

### Verificacion

- Se leyo el summary guardado del experimento bf32.
- Se reviso el historial de entrenamiento y validacion.
- Se compararon mejoras absolutas frente al ensemble, mejor individual previo y baseline.
- Se probo ensemble diagnostico con bf32 y modelos anteriores.

### Pendientes

- Ejecutar `05i` si se quiere probar mejora de datos con slices negativos; el notebook se ajusto para partir de la receta bf32.
- Generar visualizaciones cualitativas del modelo bf32.
- Actualizar el informe final de segmentacion Fase 2.

---

## 2026-05-15 - Visualizacion cualitativa de segmentacion CT

### Objetivo

Generar visualizaciones para comprobar visualmente si la prediccion del mejor modelo CT bf32 coincide con la mascara real.

### Acciones realizadas

- Se creo `scripts/generate_ct_segmentation_visualizations.py`.
- Se creo `notebooks/05k_ct_segmentation_visualization.ipynb`.
- Se genero una figura cualitativa con:
  - imagen CT,
  - mascara real,
  - prediccion,
  - overlay real/prediccion,
  - mapa de errores.
- Se guardaron metricas por slice de test.

### Artefactos

- Figura:
  - `results/segmentation/ct/qualitative/ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full_qualitative_grid.png`
- Metricas por slice:
  - `results/segmentation/ct/qualitative/ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full_test_slice_metrics.csv`
- Ejemplos seleccionados:
  - `results/segmentation/ct/qualitative/ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full_selected_examples.csv`

### Interpretacion visual

- Overlay:
  - verde = mascara real,
  - rojo = prediccion,
  - superposicion indica coincidencia entre ambas.
- Mapa de errores:
  - verde = verdadero positivo,
  - rojo = falso positivo,
  - azul = falso negativo.
- Los peores ejemplos muestran lesiones muy pequenas donde el modelo falla por falsos positivos o falsos negativos.
- Los mejores ejemplos muestran buena correspondencia anatomica con Dice alto.

### Verificacion

- El script compila correctamente.
- El notebook `05k_ct_segmentation_visualization.ipynb` es JSON valido.
- La figura se genero correctamente y fue revisada visualmente.
- Se creo y verifico `scripts/run_ct_segmentation_visualizations.sh` como lanzador simple para regenerar la visualizacion cualitativa del mejor modelo CT.
- Se amplio el script para listar/generar visualizaciones separadas por cada experimento CT con `scripts/run_ct_segmentation_visualizations.sh all`.
- Se corrigio la carga de modelos CT 2.5D en `scripts/generate_ct_segmentation_visualizations.py`: ahora el script infiere `in_channels=3` cuando el summary tiene `ct_context_slices=true` o, como respaldo, desde el checkpoint.
- Se verifico el experimento `ct_attention_unet_ct25d_mixed50_patch160_pos80_tversky_pos20_thr_segmentation` en una carpeta temporal; el checkpoint carga correctamente y genera metricas, ejemplos y figura cualitativa.

---

## 2026-05-17 - Inicio Fase 3: explicabilidad Grad-CAM

### Objetivo

Pasar de segmentacion a explicabilidad, usando los modelos finales de clasificacion y las mascaras disponibles para cuantificar si las explicaciones se concentran en regiones anatomicas o patologicas.

### Acciones realizadas

- Se implemento `src/evaluation/explainability.py` con:
  - Grad-CAM propio en PyTorch,
  - seleccion de capas objetivo para ResNet-50, DenseNet-121 y EfficientNet-B0,
  - carga de checkpoints de clasificacion,
  - binarizacion de saliencia,
  - IoU saliencia-mascara,
  - ratio de saliencia dentro de mascara,
  - indicador de pico maximo dentro de mascara.
- Se creo `scripts/generate_xai_explanations.py` para generar figuras y metricas XAI.
- Se creo `scripts/run_xai_explainability.sh` como lanzador reproducible.
- Se creo `notebooks/06_explainability.ipynb`.
- Se creo `docs/informe_explicabilidad_fase3.md`.

### Artefactos generados

- CXR:
  - `results/explainability/cxr/cxr_densenet121_weighted_ce_full/`
- CT:
  - `results/explainability/ct/ct_densenet121_baseline_full/`

### Resultados iniciales

| Modalidad | Modelo | N | IoU saliencia-mascara | Ratio saliencia dentro mascara | Pico dentro mascara |
|---|---|---:|---:|---:|---:|
| CXR | `cxr_densenet121_weighted_ce_full` | 12 | 0.2255 | 0.3143 | 0.3333 |
| CT | `ct_densenet121_baseline_full` | 2 | 0.0146 | 0.0062 | 0.0000 |

### Interpretacion

- En CXR, Grad-CAM muestra alineacion parcial con las mascaras pulmonares. Esto solo mide plausibilidad anatomica, no localizacion de lesion COVID.
- En CT, Grad-CAM no se alinea bien con las mascaras de infeccion del split de test. Es un hallazgo negativo relevante: el clasificador puede acertar la clase, pero su explicacion no demuestra atencion localizada en la lesion.

### Verificacion

- `src/evaluation/explainability.py` y `scripts/generate_xai_explanations.py` compilan correctamente.
- `scripts/run_xai_explainability.sh` pasa validacion de sintaxis bash.
- `notebooks/06_explainability.ipynb` es JSON valido.
- Se corrigieron saltos de linea literales `\n` en `notebooks/06_explainability.ipynb`; las celdas de codigo parsean correctamente.
- Se ejecuto `scripts/run_xai_explainability.sh both 2 1` correctamente.

### Pendientes

- Probar CT con `--ct-mask-split all` para aumentar ejemplos cualitativos anotados.
- Repetir CT con `ct_resnet50_baseline_full` para comparar explicabilidad del modelo con mejor accuracy.

### Ampliacion tras ejecutar notebook 06

- Se ajusto `scripts/generate_xai_explanations.py` para que las salidas CT incluyan el split de mascara en el directorio (`test` o `all`) y evitar sobrescrituras.
- Se genero CT DenseNet con todas las mascaras anotadas:
  - `results/explainability/ct/ct_densenet121_baseline_full_all/`
- Se genero CT ResNet-50 con todas las mascaras anotadas:
  - `results/explainability/ct/ct_resnet50_baseline_full_all/`
- Se regenero CT DenseNet test con carpeta explicita:
  - `results/explainability/ct/ct_densenet121_baseline_full_test/`

| Modalidad | Modelo | Split mascara | N | IoU saliencia-mascara | Ratio saliencia dentro mascara | Pico dentro mascara |
|---|---|---|---:|---:|---:|---:|
| CT | `ct_densenet121_baseline_full` | test | 2 | 0.0146 | 0.0062 | 0.0000 |
| CT | `ct_densenet121_baseline_full` | all | 4 | 0.0133 | 0.0063 | 0.0000 |
| CT | `ct_resnet50_baseline_full` | all | 4 | 0.0133 | 0.0083 | 0.0000 |

Conclusion ampliada: la baja alineacion Grad-CAM vs lesion CT no depende solo del modelo DenseNet ni del split test; tambien se observa con ResNet-50 y con todas las mascaras anotadas disponibles para el analisis cualitativo.

### Decision de alcance

Se cierra la fase XAI con Grad-CAM como metodo principal. No se incorporan LIME/SHAP en el alcance final porque:

- Grad-CAM ya responde la pregunta practica de alineacion visual con mascaras disponibles.
- LIME/SHAP implican dependencias y coste computacional adicionales.
- La conclusion principal, especialmente en CT, ya es clara: la saliencia del clasificador no se alinea bien con las mascaras de infeccion.

Siguiente fase: preparar `notebooks/07_final_analysis.ipynb` para consolidar clasificacion, segmentacion y XAI.

---

## 2026-05-17 - Preparacion Fase 4: analisis final

### Objetivo

Preparar la integracion final del TFM con tablas maestras, figuras publicables y respuestas explicitas a las preguntas de investigacion.

### Acciones realizadas

- Se creo `scripts/build_final_analysis.py`.
- Se creo `notebooks/07_final_analysis.ipynb`.
- Se creo `docs/informe_fase4_analisis_final.md`.
- Se genero la carpeta `results/final_analysis/` con:
  - resultados completos de clasificacion con intervalos bootstrap,
  - mejores modelos por accuracy y F1,
  - comparaciones McNemar aproximadas entre top-2 por modalidad,
  - resultados completos de segmentacion,
  - resultados Grad-CAM,
  - figuras finales,
  - resumen por RQ.

### Artefactos clave

- `results/final_analysis/classification_results_with_ci.csv`
- `results/final_analysis/classification_mcnemar_top2.csv`
- `results/final_analysis/segmentation_results.csv`
- `results/final_analysis/xai_gradcam_results.csv`
- `results/final_analysis/rq_summary.md`
- `results/final_analysis/figures/classification_accuracy_f1_macro.png`
- `results/final_analysis/figures/segmentation_dice_iou.png`
- `results/final_analysis/figures/xai_gradcam_alignment.png`

### Resultados principales

- Mejor clasificacion CXR: `cxr_densenet121_weighted_ce`, accuracy `0.9496`, F1-macro `0.9567`, AUC macro `0.9931`.
- Mejor clasificacion CT por accuracy: `ct_resnet50_baseline`, accuracy `0.6484`.
- Mejor clasificacion CT por F1/AUC: `ct_densenet121_baseline`, F1-macro `0.4173`, AUC macro `0.7229`.
- Mejor segmentacion CXR: `cxr_attention_unet_segmentation`, Dice `0.9853`, IoU `0.9715`.
- Mejor segmentacion CT: `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`, Dice `0.5637`, IoU `0.4305`.
- Grad-CAM CXR: IoU saliencia-mascara pulmonar `0.2255`.
- Grad-CAM CT: IoU saliencia-lesion maximo observado `0.0146`.

### Verificacion

- `scripts/build_final_analysis.py` compila correctamente.
- `notebooks/07_final_analysis.ipynb` es JSON valido.
- Todas las celdas de codigo de `07_final_analysis.ipynb` parsean correctamente.
- El script `scripts/build_final_analysis.py` se ejecuto correctamente y genero los artefactos esperados.
- Las tres figuras finales se renderizaron correctamente.

### Siguiente paso

Ejecutar visualmente `notebooks/07_final_analysis.ipynb` y usar `results/final_analysis/rq_summary.md` como base para redactar resultados/discusion.

---

## 2026-05-17 - Preparacion del marco teorico y bibliografia base

### Objetivo

Preparar material de apoyo para redactar el Estado del Arte, la Metodologia y la Discusion del TFM con conceptos tecnicos usados durante el proyecto y referencias bibliograficas verificables.

### Acciones realizadas

- Se creo `docs/marco_teorico_conceptos_y_bibliografia.md`.
- Se creo `docs/bibliografia_base_tfm.bib`.
- Se revisaron referencias web primarias/oficiales para:
  - datasets CXR/CT,
  - arquitecturas CNN,
  - transfer learning,
  - segmentacion U-Net/Attention U-Net,
  - Tversky/Focal Loss/AdamW,
  - Grad-CAM,
  - metricas de evaluacion.
- Se corrigio la referencia CXR para separar:
  - COVID-19 Radiography Database como dataset,
  - el trabajo de Rahman et al. sobre CXR,
  - el trabajo de Tahir et al. sobre localizacion y severidad en CXR.

### Artefactos clave

- `docs/marco_teorico_conceptos_y_bibliografia.md`
- `docs/bibliografia_base_tfm.bib`

### Resultado

El documento contiene explicaciones desde conceptos basicos hasta decisiones concretas del proyecto: CXR, CT, etiquetas, mascaras, splits, normalizacion, windowing HU, CNN, transfer learning, ResNet, DenseNet, EfficientNet, desbalanceo, losses, optimizacion, metricas de clasificacion, metricas de segmentacion, Grad-CAM, sesgos, overfitting y lectura honesta de resultados negativos.

### Siguiente paso

Usar este material como base para redactar el capitulo de Estado del Arte y despues transformar las secciones mas metodologicas en texto de Metodologia/Resultados.

---

## 2026-05-17 - Ampliacion de conceptos metodologicos usados

### Objetivo

Revisar si el marco teorico cubria todos los conceptos tecnicos realmente usados en el proyecto, especialmente estrategias de balanceo y mejoras experimentales de segmentacion CT.

### Acciones realizadas

- Se revisaron `src/training/classification_experiment.py`, `src/training/segmentation_experiment.py`, `src/data/segmentation.py`, `src/data/datasets.py`, `src/data/ct_preprocessing.py` y documentos de resultados.
- Se amplio `docs/marco_teorico_conceptos_y_bibliografia.md` con:
  - split estratificado,
  - fusion CT-3/CT-4 como `CT-3+`,
  - extraccion de slices CT,
  - modo smoke/full,
  - reproducibilidad,
  - baseline sin balanceo,
  - pesos por frecuencia inversa,
  - `WeightedRandomSampler`,
  - diferencias entre weighted CE, focal loss y oversampling,
  - desbalance pixel a pixel,
  - `pos_weight`,
  - logits, softmax y sigmoid,
  - entrenamiento en dos fases,
  - hardware CPU/GPU/MPS,
  - encoder-decoder y skip connections,
  - busqueda de threshold en validacion,
  - componentes conectados,
  - patch-based training,
  - positive crop sampling,
  - mixed context training,
  - 2.5D,
  - slices negativos,
  - capacidad `base_features`,
  - ensemble por promedio de probabilidades,
  - ablation study.
- Se amplio `docs/bibliografia_base_tfm.bib` con referencias oficiales de PyTorch para `WeightedRandomSampler`, `BCEWithLogitsLoss`, `AdamW` y `ReduceLROnPlateau`.

### Resultado

El marco teorico queda mas alineado con la metodologia real del TFM y ya no se limita a conceptos generales de deep learning. Ahora tambien documenta las decisiones experimentales que explican la evolucion de los resultados, especialmente en segmentacion CT.

---

## 2026-05-20 - Guia de lectura por concepto y bibliografia comentada

### Objetivo

Profundizar los conceptos tecnicos del TFM con bibliografia web, papers y documentacion oficial, para facilitar la lectura previa y la redaccion del Estado del Arte, Metodologia y Discusion.

### Acciones realizadas

- Se creo `docs/guia_lectura_conceptos_tfm.md`.
- Se estructuro la guia por concepto con:
  - concepto resumido,
  - lecturas prioritarias,
  - parrafo para entender,
  - como llevarlo a la memoria.
- Se cubrieron 29 bloques conceptuales:
  - COVID-19, CXR y CT,
  - datasets,
  - shortcut learning,
  - HU/windowing/NIfTI,
  - splits y leakage,
  - augmentation,
  - CNN/transfer learning,
  - ResNet/DenseNet/EfficientNet,
  - desbalanceo,
  - weighted CE/focal/oversampling,
  - losses y optimizacion,
  - metricas,
  - bootstrap/McNemar,
  - segmentacion,
  - U-Net/Attention U-Net,
  - Dice/IoU/Tversky,
  - threshold/postprocesado,
  - patches/positive crops,
  - mixed context/2.5D/slices negativos,
  - capacidad/ablations,
  - ensemble,
  - Grad-CAM,
  - LIME/SHAP,
  - resultados negativos.
- Se amplio `docs/bibliografia_base_tfm.bib` con nuevas referencias para augmentation, class imbalance, metricas de segmentacion, Generalised Dice, deep ensembles, McNemar, NIfTI, Hounsfield Units, windowing CT y 2.5D.
- Se enlazo la nueva guia desde `docs/marco_teorico_conceptos_y_bibliografia.md`.

### Artefactos clave

- `docs/guia_lectura_conceptos_tfm.md`
- `docs/bibliografia_base_tfm.bib`
- `docs/marco_teorico_conceptos_y_bibliografia.md`

### Resultado

La preparacion teorica queda separada en dos capas: un marco teorico base y una guia de lectura profunda. La guia permite estudiar cada concepto y convertirlo en parrafos redactables con bibliografia asociada.

---

## 2026-05-20 - Borrador de introduccion del TFM

### Objetivo

Preparar una introduccion academica para la memoria del TFM, apoyada en datos reales de los dos datasets y bibliografia relevante.

### Acciones realizadas

- Se verifico que el TFM usa dos datasets principales:
  - COVID-19 Radiography Database para CXR.
  - MosMedData para CT.
- Se recopilaron cifras clave:
  - CXR: 21.165 imagenes en cuatro clases: COVID-19, Normal, Lung Opacity y Viral Pneumonia.
  - CT: 1.110 estudios CT en MosMedData, con clases CT-0 a CT-4 y 50 estudios con mascaras de infeccion.
- Se revisaron referencias para justificar:
  - papel de la imagen toracica en COVID-19,
  - datasets CXR/CT,
  - deep learning en clasificacion,
  - segmentacion medica,
  - explicabilidad y shortcut learning.
- Se creo `docs/introduccion_tfm_borrador.md`.

### Artefacto clave

- `docs/introduccion_tfm_borrador.md`

### Resultado

El documento contiene:

- datos base de los dos datasets,
- referencias recomendadas,
- introduccion larga lista para adaptar a la memoria,
- version breve alternativa,
- ideas fuertes para defender,
- citas recomendadas para insertar.

---

## 2026-05-22 - Notebook 08: calibracion e incertidumbre

### Objetivo

Crear una fase adicional ligera para analizar si la confianza probabilistica de los clasificadores CNN refleja su fiabilidad real, sin reentrenar modelos.

### Acciones realizadas

- Se creo `scripts/build_calibration_analysis.py`.
- Se creo `notebooks/08_calibration_analysis.ipynb`.
- Se creo `docs/informe_calibracion_fase5.md`.
- Se genero `results/calibration/` con tablas y figuras.
- El analisis lee los CSV de predicciones ya existentes y calcula:
  - confianza maxima media,
  - ECE,
  - MCE,
  - Brier score multiclase,
  - negative log-likelihood,
  - errores de alta confianza.

### Artefactos clave

- `results/calibration/calibration_metrics.csv`
- `results/calibration/calibration_bins.csv`
- `results/calibration/high_confidence_errors_top100.csv`
- `results/calibration/calibration_summary.md`
- `results/calibration/figures/reliability_diagrams_selected_models.png`
- `results/calibration/figures/confidence_histograms_selected_models.png`
- `results/calibration/figures/ece_by_experiment.png`

### Resultados iniciales

- CXR mejor modelo final `cxr_densenet121_weighted_ce`: ECE `0.0109`, Brier `0.0801`.
- CT mejor accuracy `ct_resnet50_baseline`: ECE `0.0409`, Brier `0.4870`, `42` errores con confianza `>= 0.90`.
- CT mejor F1/AUC `ct_densenet121_baseline`: ECE `0.0214`, Brier `0.4844`, `14` errores con confianza `>= 0.90`.

### Verificacion

- `scripts/build_calibration_analysis.py` compila correctamente.
- `notebooks/08_calibration_analysis.ipynb` es JSON valido.
- Todas las celdas de codigo de `08_calibration_analysis.ipynb` parsean correctamente.
- El script `scripts/build_calibration_analysis.py` se ejecuto correctamente y genero los artefactos esperados.

---

## 2026-05-30 - Revision de version principal del TFM en PDF

### Objetivo

Revisar la version principal del documento del TFM (`main (2).pdf`) y comprobar si el texto escrito coincide con el trabajo experimental realmente realizado.

### Acciones realizadas

- Se extrajo el texto del PDF `main (2).pdf` para revisar estructura y contenido.
- Se compararon introduccion, objetivos, estado del arte, metodologia y resultados contra la bitacora y los informes del proyecto.
- Se verifico que el texto no afirma como implementados metodos que no se usaron:
  - no se presenta SHAP/LIME como experimento propio,
  - no se presenta ResUNet como arquitectura usada,
  - no se presenta 3D U-Net como metodologia propia.
- Se detecto que metodologia, resultados, discusion y conclusiones estan todavia incompletos o como esqueleto.
- Se creo un informe de revision con correcciones prioritarias y estructura recomendada.

### Artefacto creado

- `docs/revision_main_2_pdf.md`

### Resultado

La introduccion, objetivos y estado del arte estan alineados en lo esencial con el TFM. La version no esta lista para entrega porque faltan las secciones metodologicas y de resultados con cifras reales, ademas de la discusion y conclusiones. Tambien hay que corregir espacios, acentos y frases pegadas producidas por comandos LaTeX o formato.

---

## 2026-05-30 - Mejora de trabajos relacionados y gaps

### Objetivo

Revisar si el apartado de trabajos relacionados y gaps identificados puede mejorar en claridad, foco academico y coherencia con el TFM.

### Acciones realizadas

- Se comparo el apartado actual con `docs/comparativa_trabajos_relacionados.md`.
- Se concluyo que la idea general es correcta, pero la tabla actual puede resultar demasiado densa.
- Se propuso una version mas narrativa:
  - organizar trabajos por lineas de investigacion,
  - mantener una tabla compacta,
  - cerrar con cuatro gaps metodologicos claros.
- Se remarco que el TFM no debe defenderse como arquitectura nueva, sino como integracion reproducible de clasificacion, segmentacion, Grad-CAM, desbalanceo y calibracion.

### Artefacto creado

- `docs/propuesta_mejora_trabajos_relacionados_gaps.md`

### Resultado

La seccion no sobra, pero conviene simplificarla. La recomendacion es reducir la matriz de checklist, evitar demasiados simbolos y reforzar una narrativa academica centrada en la integracion de dimensiones que suelen aparecer separadas en la literatura.

---

## 2026-05-31 - Estructura de metodologia del TFM

### Objetivo

Definir como dividir el capitulo de metodologia para documentar de forma reproducible el pipeline experimental realizado.

### Acciones realizadas

- Se propuso una metodologia dividida en nueve partes:
  - diseno general del estudio,
  - conjuntos de datos,
  - preprocesamiento y particiones,
  - clasificacion,
  - segmentacion,
  - explicabilidad Grad-CAM,
  - calibracion probabilistica,
  - metricas de evaluacion,
  - diseno experimental y reproducibilidad.
- Se remarco que la metodologia debe explicar lo ejecutado, no repetir el estado del arte.
- Se incluyeron advertencias de coherencia:
  - CXR segmenta pulmones, no lesiones,
  - CT segmenta lesion/infeccion,
  - 2.5D fue una exploracion,
  - Grad-CAM fue el unico metodo XAI implementado,
  - la memoria debe reportar solo resultados `full`.

### Artefacto creado

- `docs/estructura_metodologia_tfm.md`

### Resultado

Queda definida una estructura base para redactar el capitulo 3 de metodologia y evitar mezclar teoria, resultados y decisiones experimentales.

### Ajuste posterior

El usuario aclara que para la memoria quiere documentar unicamente ejecuciones `full`. Por tanto, la metodologia no debe presentarse como una comparacion `smoke/full`; se describen solo los experimentos completos y sus evaluaciones finales.

---

## 2026-06-05 - Borrador redactado de metodologia

### Objetivo

Redactar una primera version completa de la seccion de metodologia basada en el pipeline realmente ejecutado.

### Acciones realizadas

- Se revisaron configuraciones, codigo y artefactos para documentar:
  - datasets CXR y CT,
  - particiones train/val/test,
  - preprocesamiento,
  - arquitecturas de clasificacion,
  - estrategias de desbalanceo,
  - segmentacion U-Net y Attention U-Net,
  - variantes CT,
  - Grad-CAM,
  - calibracion probabilistica,
  - metricas y reproducibilidad.
- Se verificaron tamanos de particion:
  - CXR clasificacion: `14815/3175/3175`.
  - CT clasificacion: `19456/4170/4155` slices y `777/166/167` estudios.
  - CT segmentacion: `508/86/110` slices y `35/7/8` estudios.
- Se mantuvieron advertencias de coherencia:
  - CXR segmenta pulmones, no lesiones COVID,
  - CT segmenta lesion/infeccion,
  - Grad-CAM es el unico metodo XAI implementado,
  - 2.5D se documenta como exploracion,
  - `bf32` corresponde a `base_features=32`.

### Artefacto creado

- `docs/borrador_metodologia_tfm.md`

### Resultado

Queda disponible un borrador casi listo para adaptar a LaTeX, con texto metodologico y una tabla final de decisiones verificadas.

---

## 2026-06-05 - Version detallada y pedagogica de metodologia

### Objetivo

Ampliar el borrador de metodologia para que explique con mas detalle y de forma facil de entender cada decision experimental.

### Acciones realizadas

- Se creo una version mas extensa de metodologia con:
  - explicaciones sencillas de cada bloque,
  - tablas de decisiones,
  - texto listo para memoria,
  - justificacion de por que se usa cada estrategia,
  - advertencias sobre interpretaciones incorrectas.
- Se incluyeron detalles de:
  - datasets,
  - splits,
  - preprocesamiento,
  - augmentacion,
  - transfer learning,
  - fine-tuning,
  - desbalanceo,
  - segmentacion,
  - variantes CT,
  - Grad-CAM,
  - calibracion,
  - metricas,
  - reproducibilidad.

### Artefacto creado

- `docs/borrador_metodologia_tfm_detallado.md`

### Resultado

La metodologia queda preparada en una version mas didactica, adecuada para explicar el trabajo en la memoria y tambien para defender oralmente las decisiones principales.

---

## 2026-06-05 - Explicacion pedagogica dentro de notebooks

### Objetivo

Anadir explicaciones dentro de cada notebook para que el codigo sea entendible desde cero: que hace cada celda, por que se ejecuta, que resultado se espera y como interpretar cada linea de Python.

### Acciones realizadas

- Se anadieron guias automaticas antes de cada celda de codigo en los 18 notebooks del proyecto.
- Cada guia incluye:
  - objetivo practico de la celda,
  - justificacion metodologica dentro del TFM,
  - resultado esperado al ejecutar,
  - explicacion linea a linea del codigo.
- Se creo el script reutilizable `scripts/add_notebook_code_explanations.py`.
- El script es idempotente:
  - elimina explicaciones generadas previamente,
  - vuelve a insertarlas actualizadas,
  - evita duplicados si se ejecuta mas de una vez.
- Se ampliaron explicaciones especificas para hiperparametros y objetos frecuentes:
  - `dataset_name`,
  - `architecture`,
  - `run_mode`,
  - `batch_size`,
  - `epochs`,
  - `learning_rate`,
  - `pos_weight`,
  - `dice_weight`,
  - `tversky_alpha`,
  - `tversky_beta`,
  - `positive_crop_prob`,
  - `threshold`,
  - `base_features`,
  - `train_df`, `val_df`, `test_df`,
  - objetos de metricas, graficas, Grad-CAM, calibracion y segmentacion.

### Verificacion

- Se valido que los 18 notebooks siguen siendo JSON valido.
- Se comprobaron 113 celdas de codigo documentadas.
- Se comprobo que cada notebook tiene una guia general de lectura.
- Se comprobo que cada celda de codigo tiene una guia explicativa.
- Se comprobo que no quedan parametros con explicacion generica.

### Artefactos modificados

- `notebooks/*.ipynb`
- `scripts/add_notebook_code_explanations.py`
- `docs/bitacora_tfm.md`

### Resultado

Los notebooks quedan preparados como material de estudio y defensa: no solo ejecutan los experimentos, sino que explican el flujo del codigo y la razon metodologica de cada paso.

---

## 2026-06-05 - Verificacion web de fiabilidad de datasets

### Objetivo

Comprobar si los datasets locales usados en el TFM concuerdan con las fuentes publicas disponibles en web.

### Acciones realizadas

- Se verifico la COVID-19 Radiography Database frente a Kaggle y articulos relacionados.
- Se verifico MosMedData frente a arXiv, Academic Torrents y articulos que resumen su distribucion.
- Se compararon los conteos locales con los conteos declarados por las fuentes.
- Se comprobaron correspondencias imagen-mascara en CXR.
- Se comprobo la correspondencia normalizada imagen-mascara en CT segmentacion.
- Se diferenciaron:
  - estudios CT originales,
  - slices 2D procesados,
  - mascaras pulmonares CXR,
  - mascaras de infeccion CT.

### Resultado

- CXR local coincide con la fuente: 21165 imagenes, 21165 mascaras, cuatro clases.
- CT local coincide con MosMedData: 1110 estudios con distribucion `254/684/125/45/2`.
- El subconjunto CT de segmentacion contiene 50 estudios con mascaras y 704 pares procesados imagen-mascara.
- No se detectaron contradicciones relevantes.

### Artefacto creado

- `docs/verificacion_fiabilidad_datasets_web.md`

---

## 2026-06-05 - Revision cientifica transversal de notebooks

### Objetivo

Revisar todos los notebooks del TFM para comprobar si el flujo experimental es defendible cientificamente, si las conclusiones estan justificadas por los experimentos y si existe algun punto metodologico que pueda interpretarse como fuga de informacion, comparacion injusta o afirmacion excesiva.

### Acciones realizadas

- Se reviso la secuencia completa de notebooks desde EDA hasta calibracion.
- Se contrastaron notebooks con los modulos de datos, entrenamiento, segmentacion, explicabilidad y analisis final.
- Se identificaron puntos fuertes:
  - split CT por `study_id`,
  - resultados finales filtrados en modo `full`,
  - uso de baselines y ablationes,
  - seleccion de umbrales de segmentacion sobre validacion en los entrenamientos principales,
  - uso de metricas adecuadas para clasificacion, segmentacion, explicabilidad y calibracion.
- Se identificaron matices metodologicos que deben documentarse:
  - `05e` debe tratarse como analisis exploratorio si el barrido de postprocesado se hizo sobre test,
  - la segmentacion CT se evalua principalmente sobre slices positivos,
  - la clasificacion CT es por slice aunque el split sea por estudio,
  - Grad-CAM es una auditoria exploratoria y no una validacion causal,
  - los notebooks que conservan configuraciones `smoke` no deben mezclarse con resultados finales.

### Resultado

El flujo experimental es defendible como estudio interno comparativo y reproducible, siempre que la memoria explique correctamente las limitaciones y no convierta analisis exploratorios en conclusiones finales.

### Artefacto creado

- `docs/revision_cientifica_notebooks.md`

---

## 2026-06-05 - Unificacion de ejecuciones en modo full

### Objetivo

Eliminar el modo `smoke` del flujo ejecutable y dejar los notebooks/scripts principales configurados unicamente para experimentos `full`.

### Acciones realizadas

- Se cambio `src/training/classification_experiment.py` para que `make_run_config` acepte solo `run_mode='full'`.
- Se cambio `src/training/segmentation_experiment.py` para que `make_segmentation_run_config` y `limit_segmentation_samples` acepten solo `run_mode='full'`.
- Se actualizo `notebooks/02_cxr_classification.ipynb`:
  - `RUN_MODE = 'full'`;
  - matriz directa de 3 arquitecturas x 3 estrategias;
  - reutilizacion de artefactos full existentes;
  - eliminacion del entrenamiento de muestra reducido.
- Se actualizo `notebooks/03_ct_classification.ipynb`:
  - arquitecturas y estrategias full definidas directamente;
  - eliminacion de ramas condicionales por modo.
- Se actualizo `notebooks/04_classification_results.ipynb`:
  - carga exclusiva de `*_full_summary.json`;
  - mensaje de error apuntando a `scripts/run_phase1_full.py`;
  - notas metodologicas centradas en resultados full.
- Se actualizo `notebooks/05_segmentation.ipynb` para describir solo ejecuciones full.
- Se elimino `scripts/run_phase1_smoke.py`.
- Se creo `scripts/run_phase1_full.py`.
- Se eliminaron artefactos antiguos `*_smoke.*` de `results/` y `models/`.
- Se regeneraron las explicaciones automaticas de codigo en notebooks para evitar referencias obsoletas.
- Se actualizaron documentos vivos de metodologia/revision para reflejar el estado actual.

### Verificacion

- `rg` no encuentra referencias a `smoke` en `notebooks`, `scripts` ni `src`.
- `find results models -name '*smoke*'` no devuelve artefactos antiguos.
- Todos los notebooks son JSON valido.
- Compilan sin errores:
  - `src/training/classification_experiment.py`,
  - `src/training/segmentation_experiment.py`,
  - `scripts/run_phase1_full.py`,
  - `scripts/build_final_analysis.py`,
  - `scripts/build_calibration_analysis.py`.

### Resultado

El flujo ejecutable del TFM queda centrado en experimentos completos. Las menciones antiguas a `smoke` solo permanecen como historial en la bitacora, no como parte del pipeline actual.

---

## 2026-06-07 - Evaluacion CT por estudio

### Objetivo

Complementar la evaluacion CT por slice con una evaluacion por `study_id`, ya que MosMedData asigna la etiqueta de severidad al estudio completo y no a cada corte individual.

### Acciones realizadas

- Se creo `scripts/build_ct_study_level_analysis.py`.
- Se reconstruyo el test split CT con `get_ct_dataframes` y la semilla del proyecto.
- Se unieron las predicciones CT slice-level existentes con `study_id`, verificando que `y_true` coincidiera antes de agregar.
- Se probaron varias estrategias de agregacion por estudio:
  - promedio de probabilidades,
  - promedio ponderado por confianza,
  - maximo por clase,
  - top-3 por clase,
  - top-20% por clase,
  - voto mayoritario.
- Se generaron metricas, predicciones agregadas, matrices de confusion, informes por clase y figuras.
- Se creo `notebooks/09_ct_study_level_analysis.ipynb` como fase ligera de analisis, sin reentrenamiento.

### Resultado

La mejor configuracion por estudio fue `ct_resnet50_weighted_ce` con agregacion `mean_probability`, alcanzando accuracy 0.6108, F1-macro 0.5164 y AUC-ROC macro 0.7819 sobre 167 estudios de test. La comparacion apoya que la evaluacion por estudio es mas coherente con la etiqueta original de MosMedData y reduce el impacto de slices aislados poco informativos.

### Artefactos creados

- `results/classification/ct_study_level/ct_study_level_metrics.csv`
- `results/classification/ct_study_level/ct_study_level_predictions.csv`
- `results/classification/ct_study_level/ct_slice_predictions_with_study.csv`
- `results/classification/ct_study_level/ct_study_level_confusion_matrices.csv`
- `results/classification/ct_study_level/ct_study_level_classification_reports.csv`
- `results/classification/ct_study_level/ct_study_level_summary.md`
- `results/classification/ct_study_level/figures/ct_slice_vs_study_f1_macro.png`
- `results/classification/ct_study_level/figures/ct_study_level_aggregation_comparison.png`
- `results/classification/ct_study_level/figures/ct_study_level_best_confusion_matrix.png`

---

## 2026-06-21 - Figuras comparativas de matrices de confusion CXR

### Objetivo

Crear una visualizacion compacta para comparar las matrices de confusion de los nueve clasificadores CXR entrenados en modo full.

### Acciones realizadas

- Se generó una figura 3x3 con las matrices de confusion absolutas de CXR.
- Se generó una segunda figura 3x3 normalizada por clase real, mas adecuada para comparar errores relativos entre clases con distinto soporte.
- Las figuras se construyeron a partir de los CSV `*_confusion_matrix.csv` ya existentes, sin reentrenar modelos.

### Resultado

Las figuras permiten observar de forma rapida que la mayoria de modelos tienen buen comportamiento global y que el principal patron de error se concentra entre `Lung Opacity` y `Normal`.

### Artefactos creados

- `results/classification/cxr/figures/cxr_all_confusion_matrices_grid.png`
- `results/classification/cxr/figures/cxr_all_confusion_matrices_grid_normalized.png`

---

## 2026-06-21 - Figuras comparativas de matrices de confusion CT

### Objetivo

Crear una visualizacion equivalente a CXR para comparar las matrices de confusion de los nueve clasificadores CT por slice entrenados en modo full.

### Acciones realizadas

- Se generó una figura 3x3 con matrices de confusion absolutas de CT por slice.
- Se generó una figura 3x3 normalizada por clase real, mas informativa para analizar el desbalanceo entre CT-0, CT-1, CT-2 y CT-3+.
- Las figuras se construyeron a partir de los CSV `*_confusion_matrix.csv` existentes, sin reentrenar modelos.

### Resultado

Las figuras muestran que los modelos baseline tienden a concentrar predicciones en la clase mayoritaria CT-1, mientras que las perdidas ponderadas reparten mas las predicciones y recuperan parte del recall de CT-2 y CT-3+, aunque con menor accuracy global.

### Artefactos creados

- `results/classification/ct/figures/ct_all_confusion_matrices_grid.png`
- `results/classification/ct/figures/ct_all_confusion_matrices_grid_normalized.png`

---

## 2026-06-22 - Fase 10: seleccion de slices informativos en CT

### Objetivo

Estudiar si la clasificacion CT mejora al reducir slices poco informativos. La motivacion es que MosMedData asigna la etiqueta de severidad al estudio completo, mientras que el modelo 2D se entrena con cortes individuales; por tanto, algunos slices pueden contener poca informacion diagnostica para la etiqueta global del volumen.

### Acciones realizadas

- Se creo `src/data/ct_slice_selection.py` para calcular metricas simples por slice:
  - posicion normalizada dentro del volumen,
  - intensidad media,
  - desviacion estandar,
  - proporcion de pixeles no cero,
  - proporcion de pixeles por encima de 5, 10 y 25.
- Se creo `scripts/run_ct_informative_slice_experiments.py` para preparar variantes y entrenar experimentos CT full sobre metadata filtrada.
- Se creo `scripts/build_ct_informative_slice_analysis.py` para comparar resultados por slice y por estudio cuando existan modelos entrenados.
- Se creo `notebooks/10_ct_informative_slice_selection.ipynb` como fase documentada para preparar, lanzar y analizar este experimento.
- Se generaron metadatos filtrados para cinco variantes:
  - `central30_70`,
  - `central35_65`,
  - `top12_tissue`,
  - `top16_tissue`,
  - `top20_tissue`.

### Decisiones tecnicas

- No se usa una limpieza ingenua de slices completamente negros, porque el analisis previo mostro que no existen cortes realmente vacios en el CT procesado: el minimo de pixeles no cero ronda el 48%.
- Se mantiene el split por `study_id` en todas las variantes para evitar fuga de informacion entre train, validacion y test.
- Las variantes propuestas conservan todos los estudios; solo reducen el numero de slices por estudio.
- La variante recomendada para el primer entrenamiento es `top16_tissue` con `resnet50` y `weighted_ce`, porque ResNet50 con perdida ponderada fue la mejor configuracion CT en evaluacion por estudio.

### Verificacion

Comandos ejecutados:

```bash
.conda/bin/python -m compileall src/data/ct_slice_selection.py scripts/run_ct_informative_slice_experiments.py scripts/build_ct_informative_slice_analysis.py
.conda/bin/python scripts/run_ct_informative_slice_experiments.py prepare
.conda/bin/python scripts/build_ct_informative_slice_analysis.py
python -m json.tool notebooks/10_ct_informative_slice_selection.ipynb
```

Resultado:
- Los scripts compilan sin errores.
- La preparacion genera los CSV de seleccion correctamente.
- El script de analisis informa de forma limpia que aun no hay modelos entrenados.
- El notebook 10 es JSON valido.

### Resultados

Todas las variantes conservan los 1110 estudios originales. La reduccion de slices queda asi:

| Variante | CT-0 | CT-1 | CT-2 | CT-3+ |
|---|---:|---:|---:|---:|
| central30_70 | 65.2% | 65.5% | 65.2% | 64.5% |
| central35_65 | 49.2% | 49.7% | 49.6% | 49.7% |
| top12_tissue | 45.9% | 48.6% | 48.4% | 49.1% |
| top16_tissue | 61.2% | 64.8% | 64.6% | 65.4% |
| top20_tissue | 76.5% | 80.9% | 80.7% | 81.8% |

### Artefactos creados

- `src/data/ct_slice_selection.py`
- `scripts/run_ct_informative_slice_experiments.py`
- `scripts/build_ct_informative_slice_analysis.py`
- `notebooks/10_ct_informative_slice_selection.ipynb`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/ct_slice_quality_features.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/ct_slice_selection_summary.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/central30_70_metadata.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/central35_65_metadata.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/top12_tissue_metadata.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/top16_tissue_metadata.csv`
- `data/MosMedData_Chest_Scan/processed_2d_slices/informative_slice_metadata/top20_tissue_metadata.csv`
- `results/classification/ct_informative_slices/ct_slice_selection_summary.csv`
- `results/classification/ct_informative_slices/figures/ct_slice_selection_keep_ratio.png`

### Pendientes

Ejecutar al menos el primer entrenamiento recomendado:

```bash
.conda/bin/python scripts/run_ct_informative_slice_experiments.py train-one --variant top16_tissue --architecture resnet50 --strategy weighted_ce
```

Despues, ejecutar:

```bash
.conda/bin/python scripts/build_ct_informative_slice_analysis.py
```

Con esos resultados se podra comparar si la seleccion de slices mejora el rendimiento CT por slice y por estudio.

### Actualizacion tras entrenamiento

Se entrenaron dos variantes con `resnet50` y `weighted_ce`:

| Experimento | Unidad | Agregacion | Accuracy | F1-macro | AUC macro |
|---|---|---|---:|---:|---:|
| top16_tissue | slice | slice | 0.5374 | 0.4247 | 0.7206 |
| top16_tissue | study | mean_probability | 0.6048 | 0.4770 | 0.7719 |
| top20_tissue | slice | slice | 0.5388 | 0.4330 | 0.7482 |
| top20_tissue | study | mean_probability | 0.6108 | 0.4990 | 0.8059 |

Comparado con `ct_resnet50_weighted_ce` sin seleccion de slices, `top20_tissue` mejora la evaluacion por slice en F1-macro (0.4330 frente a 0.4130) y AUC macro (0.7482 frente a 0.7181). En evaluacion por estudio mantiene la accuracy (0.6108), mejora el AUC macro (0.8059 frente a 0.7819), pero queda por debajo en F1-macro (0.4990 frente a 0.5164). La conclusion metodologica es que seleccionar slices mas informativos reduce ruido a nivel de corte y mejora discriminacion probabilistica, pero no resuelve por completo el desequilibrio entre clases ni la dificultad de representar la severidad de un volumen CT completo mediante slices 2D.

---

## 2026-06-23 - Auditoria de mascaras CT de segmentacion

### Objetivo

Comprobar si las mascaras CT usadas en segmentacion estaban vacias o si realmente contenian regiones positivas anotadas, ya que visualmente algunos ejemplos parecian no mostrar infeccion.

### Resultado

Se revisaron las `704` mascaras CT procesadas en `processed_segmentation_slices`. Todas contienen al menos un pixel positivo; no hay mascaras vacias. Sin embargo, las regiones anotadas son muy pequenas:

- mediana: `215.5` pixeles positivos, aproximadamente `0.329%` de una imagen `256 x 256`;
- 29.4% de las mascaras tienen `<= 100` pixeles positivos;
- 54.0% de las mascaras tienen `<= 250` pixeles positivos;
- 94.9% de las mascaras tienen `<= 1000` pixeles positivos;
- area maxima observada: `1749` pixeles, aproximadamente `2.669%`.

Todas las mascaras disponibles localmente pertenecen a estudios `CT-1`, por lo que la segmentacion CT debe interpretarse como segmentacion de afectacion leve anotada, no como segmentacion generalizable a todos los grados de severidad de MosMedData.

### Artefactos creados

- `results/segmentation/ct/mask_audit/ct_mask_area_audit.csv`
- `results/segmentation/ct/mask_audit/ct_mask_area_examples.png`

---

## 2026-06-23 - Figura cualitativa CXR de segmentacion pulmonar

### Objetivo

Crear una figura cualitativa para mostrar ejemplos de segmentacion pulmonar en CXR, complementando las metricas Dice e IoU con evidencia visual.

### Resultado

Se genero una cuadricula con imagen CXR, mascara real, prediccion de Attention U-Net y overlay. Los ejemplos muestran que la prediccion pulmonar se solapa de forma muy alta con la mascara real, coherente con Dice 0.9853 e IoU 0.9715.

### Artefactos creados

- `results/segmentation/cxr/qualitative/cxr_attention_unet_segmentation/cxr_attention_unet_segmentation_full_qualitative_grid.png`
- `results/segmentation/cxr/qualitative/cxr_attention_unet_segmentation/cxr_attention_unet_segmentation_full_selected_examples.csv`

---

## 2026-06-25 - Revision del notebook Grad-CAM

### Objetivo

Revisar `notebooks/06_explainability.ipynb` para comprobar si la fase Grad-CAM estaba metodologicamente clara y si podia mejorarse la lectura de resultados.

### Acciones realizadas

- Se hizo mas robusta la deteccion de la raiz del proyecto, usando la ruta absoluta del TFM como respaldo.
- Se actualizo `scripts/run_xai_explainability.sh` para aceptar `CT_MASK_SPLIT` como cuarto argumento.
- Se explicito en el notebook que Grad-CAM se calcula sobre la clase predicha por el modelo, porque la finalidad es explicar la decision tomada.
- Se separo la lectura de resultados por modalidad y por `mask_split`.
- Se filtro una carpeta CT antigua sin sufijo `_test`/`_all` para evitar duplicar resultados en la tabla del notebook.
- Se mejoro la visualizacion de figuras separando CXR y CT.
- Se amplio la interpretacion final del notebook, indicando que CT-test tiene pocos ejemplos por la disponibilidad limitada de mascaras.

### Verificacion

Comandos ejecutados:

```bash
python -m json.tool notebooks/06_explainability.ipynb
bash -n scripts/run_xai_explainability.sh
```

Tambien se ejecuto una lectura ligera de las celdas de setup y metricas. El notebook queda leyendo 4 summaries limpios y 22 ejemplos Grad-CAM:

- CXR DenseNet121 weighted CE test: 12 ejemplos.
- CT DenseNet121 baseline test: 2 ejemplos.
- CT DenseNet121 baseline all: 4 ejemplos.
- CT ResNet50 baseline all: 4 ejemplos.

### Resultado interpretativo

Los resultados Grad-CAM deben presentarse como auditoria visual, no como prueba causal ni segmentacion. CXR muestra alineamiento parcial con pulmones; CT muestra alineamiento muy bajo con mascaras de afectacion, coherente con la dificultad del problema, el tamano pequeno de las mascaras y el hecho de que el clasificador CT fue entrenado con etiquetas de severidad, no con supervision espacial.

---

## 2026-06-27 - Clasificacion CT a nivel de estudio mediante meta-clasificador

### Objetivo

Explorar una mejora metodologica para CT que trate la unidad de prediccion como estudio completo, no como slice aislado. La motivacion es que MosMedData define las etiquetas CT-0, CT-1, CT-2 y CT-3+ a nivel de volumen, mientras que los modelos 2D entrenados previamente reciben cortes individuales.

### Acciones realizadas

- Se creo `notebooks/12_ct_study_level_meta_classifier.ipynb`.
- Se creo `scripts/run_ct_study_level_meta_classifier.py`.
- Se reutilizo el modelo `ct_top20_tissue_resnet50_weighted_ce` como extractor de probabilidades por slice.
- Se generaron probabilidades por slice para train, validacion y test.
- Se construyeron caracteristicas por estudio usando medias, maximos, percentiles, votos, confianza y entropia de las probabilidades de los slices.
- Se entrenaron candidatos ligeros a nivel de estudio, seleccionando el mejor mediante validacion y evaluando una sola vez en test.

### Resultado

El mejor metodo fue `meta_random_forest_balanced_depth4`, con:

- accuracy test: `0.6527`;
- F1-macro test: `0.5365`;
- F1-weighted test: `0.6641`;
- AUC macro test: `0.8342`.

Este resultado mejora la agregacion simple por media de probabilidades, que obtuvo:

- accuracy test: `0.6108`;
- F1-macro test: `0.4990`;
- AUC macro test: `0.8059`.

La lectura principal es que resumir la informacion de varios slices por estudio aporta una mejora moderada y metodologicamente coherente. No sustituye a un modelo 3D, pero aproxima mejor la unidad real de etiqueta del dataset.

### Artefactos creados

- `notebooks/12_ct_study_level_meta_classifier.ipynb`
- `scripts/run_ct_study_level_meta_classifier.py`
- `results/classification/ct_study_level_meta/ct_study_level_meta_metrics.csv`
- `results/classification/ct_study_level_meta/ct_study_level_meta_validation_candidates.csv`
- `results/classification/ct_study_level_meta/ct_study_level_meta_predictions.csv`
- `results/classification/ct_study_level_meta/ct_study_level_features.csv`
- `results/classification/ct_study_level_meta/figures/ct_study_level_meta_test_metrics.png`
- `results/classification/ct_study_level_meta/figures/ct_study_level_meta_best_confusion_matrix.png`
