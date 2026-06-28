# Revision cientifica transversal de los notebooks

Fecha: 2026-06-05

## Dictamen ejecutivo

El flujo experimental del TFM es defendible cientificamente si se presenta con las limitaciones correctas. La estructura general tiene sentido: exploracion de datos, clasificacion CXR/CT, manejo del desbalanceo, segmentacion CXR/CT, ablationes especificas para mejorar CT, Grad-CAM, analisis final, calibracion y evaluacion CT por estudio.

No se observa una decision metodologica absurda o sin sentido. Tras la limpieza de modo de ejecucion, los notebooks y scripts principales quedan configurados para usar solamente resultados `full`. Lo que si aparece son varios puntos que deben documentarse con precision para evitar sobreafirmar resultados:

- el analisis de postprocesado de CT en `05e` barre parametros sobre test, por lo que debe tratarse como diagnostico exploratorio, no como optimizacion final;
- la segmentacion CT se evalua principalmente sobre slices con mascara positiva, por tanto mide delineacion de lesion cuando hay lesion visible, no deteccion en volumen completo;
- las metricas CT de clasificacion iniciales se calculan por slice, aunque el split sea por estudio; esto es correcto como evaluacion por imagen 2D, pero las inferencias estadisticas por slice pueden sobreestimar independencia;
- se ha anadido una evaluacion complementaria por `study_id`, agregando probabilidades de slices, para aproximar la unidad original de etiquetado de MosMedData;
- Grad-CAM se usa correctamente como auditoria cualitativa, pero la muestra es pequena y no permite una conclusion estadistica fuerte.

Con estos matices, el experimento queda cientificamente justificable como un estudio experimental interno, comparativo y reproducible, no como validacion clinica externa.

## Evidencia revisada

Se revisaron los notebooks principales:

- `00_eda.ipynb`
- `02_cxr_classification.ipynb`
- `03_ct_classification.ipynb`
- `04_classification_results.ipynb`
- `05_segmentation.ipynb`
- `05b_ct_tversky_variant.ipynb`
- `05c_ct_patch_training.ipynb`
- `05d_ct_mixed_context_training.ipynb`
- `05e_ct_postprocessing_analysis.ipynb`
- `05f_ct_25d_context_training.ipynb`
- `05g_ct_mixed_context_ablation.ipynb`
- `05h_ct_mixed_ensemble.ipynb`
- `05i_ct_negative_slice_training.ipynb`
- `05j_ct_high_capacity_refinement.ipynb`
- `05k_ct_segmentation_visualization.ipynb`
- `06_explainability.ipynb`
- `07_final_analysis.ipynb`
- `08_calibration_analysis.ipynb`
- `09_ct_study_level_analysis.ipynb`

Tambien se revisaron modulos de soporte:

- `src/data/preprocessing.py`
- `src/data/ct_preprocessing.py`
- `src/data/segmentation.py`
- `src/data/datasets.py`
- `src/data/transforms.py`
- `src/training/classification_experiment.py`
- `src/training/segmentation_experiment.py`
- `src/evaluation/explainability.py`
- `scripts/build_final_analysis.py`
- `scripts/build_calibration_analysis.py`
- `scripts/build_ct_study_level_analysis.py`

## Aspectos cientificamente solidos

### 1. Estructura experimental por fases

La division en fases es coherente:

1. Analisis exploratorio y verificacion de datos.
2. Clasificacion CXR y CT.
3. Comparacion de resultados de clasificacion.
4. Segmentacion CXR y CT.
5. Ablationes y mejoras especificas para CT.
6. Visualizacion cualitativa de segmentacion.
7. Explicabilidad con Grad-CAM.
8. Analisis final y calibracion.
9. Evaluacion CT por estudio.

Esto permite defender el TFM como un proceso experimental incremental: primero se establecen baselines, despues se analizan fallos, y finalmente se introducen variantes justificadas.

### 2. Separacion entrenamiento, validacion y test

En clasificacion y segmentacion, el codigo distingue conjuntos de entrenamiento, validacion y prueba. En CT, ademas, el split se hace por `study_id`, lo cual es una buena practica importante porque evita que slices del mismo estudio aparezcan en train y test al mismo tiempo.

Esto es especialmente relevante en CT, donde una division aleatoria por slice produciria fuga de informacion.

### 3. Uso correcto de metricas

Las metricas principales estan bien elegidas:

- clasificacion: accuracy, F1, AUC, matriz de confusion y metricas por clase;
- segmentacion: Dice e IoU como metricas principales;
- pixel accuracy como metrica secundaria, sabiendo que puede ser enganosa por predominio del fondo;
- calibracion: ECE, Brier score, NLL y errores de alta confianza.

La inclusion de calibracion mejora mucho la calidad metodologica porque no solo mide si el modelo acierta, sino si sus probabilidades son fiables.

### 4. Manejo del desbalanceo

La comparacion entre baseline, weighted cross-entropy, focal loss y oversampling esta cientificamente justificada. Es buena practica incluir un baseline sin correccion para comprobar si las tecnicas de balanceo aportan mejora real.

Tambien es correcto que los resultados negativos se conserven: en CT, las estrategias de balanceo no superan claramente al baseline, y eso es un resultado defendible.

### 5. Segmentacion como experimento independiente y como apoyo a interpretabilidad

El uso de U-Net y Attention U-Net es coherente con el estado del arte en segmentacion biomédica. En CXR se segmenta pulmon; en CT se segmenta lesion/infeccion.

La comparacion entre baselines y variantes de CT tiene logica experimental:

- baseline U-Net / Attention U-Net;
- perdida Tversky ponderada;
- entrenamiento por parches;
- positive crop sampling;
- mixed context training;
- 2.5D;
- ensemble;
- aumento de capacidad con `base_features=32`.

El mejor resultado CT actual, `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`, mejora claramente el baseline CT y justifica la narrativa de experimentacion.

### 6. Grad-CAM usado con prudencia

Grad-CAM se emplea como tecnica de explicabilidad post-hoc, no como segmentacion. Esto es correcto. La comparacion con mascaras puede aportar informacion, siempre que se explique bien:

- en CXR, la mascara pulmonar sirve para comprobar si la saliencia cae dentro del campo pulmonar;
- en CT, la mascara de lesion permite una comparacion mas patologica, pero el resultado observado indica que la saliencia no se alinea fuertemente con la lesion.

Esta conclusion es cientificamente interesante porque muestra una limitacion del clasificador CT.

## Hallazgos que deben corregirse o matizarse

### Alta prioridad

#### 1. `05e_ct_postprocessing_analysis.ipynb`: barrido de postprocesado sobre test

El notebook `05e` evalua combinaciones de `threshold` y `min_component_area` sobre el conjunto de test. Esto no es valido como seleccion final de hiperparametros, porque el test debe reservarse para evaluacion final.

Puede mantenerse si se etiqueta como:

- analisis exploratorio;
- diagnostico de sensibilidad;
- inspeccion posterior para entender el comportamiento del modelo.

No debe escribirse como:

- "se optimizo el postprocesado en test";
- "el mejor umbral final se selecciono con el test";
- "esta mejora representa el rendimiento final generalizable".

Correccion recomendada:

- mover la busqueda de `threshold` y `min_component_area` a validacion;
- aplicar una sola configuracion final en test;
- si no se reejecuta, declarar explicitamente que `05e` no se usa para seleccionar el modelo final.

#### 2. Segmentacion CT evaluada en slices positivos

Los experimentos de segmentacion CT usan principalmente `positive_mask_only=True`. Esto tiene sentido para entrenar y evaluar la calidad de delineacion de lesion cuando la lesion esta presente, pero cambia la interpretacion del experimento.

Conclusion valida:

> El modelo evalua la capacidad de segmentar lesiones en slices con anotacion positiva.

Conclusion que no debe afirmarse:

> El modelo detecta correctamente la ausencia de lesion en cualquier slice CT o en volumen completo.

Para afirmar deteccion en volumen completo haria falta incluir slices negativos en validacion/test y medir falsos positivos.

#### 3. CT clasificacion por slice y evaluacion complementaria por estudio

El split por `study_id` es correcto y evita fuga entre train/test. Sin embargo, las metricas finales se calculan por slice. Esto significa que las muestras del test no son completamente independientes, porque varios slices proceden del mismo estudio.

La evaluacion por slice es valida si se formula asi:

> Clasificacion 2D por slice CT, con particion agrupada por estudio.

Pero las pruebas estadisticas como McNemar o intervalos bootstrap por muestra deben interpretarse con cautela, porque asumen independencia entre ejemplos.

Correccion implementada:

- se creo `09_ct_study_level_analysis.ipynb`;
- se agregaron probabilidades por `study_id` mediante varias estrategias;
- se calcularon accuracy, F1 y AUC por estudio;
- el mejor resultado por estudio fue `ct_resnet50_weighted_ce` con `mean_probability`, F1-macro 0.5164 sobre 167 estudios de test.

La evaluacion por slice sigue siendo util para analizar comportamiento 2D, pero la evaluacion por estudio debe destacarse como lectura mas alineada con la etiqueta original de MosMedData.

### Prioridad media

#### 4. Configuracion full unificada

El problema detectado originalmente en `02_cxr_classification.ipynb` y `04_classification_results.ipynb` fue corregido. El notebook CXR queda configurado en `RUN_MODE = 'full'`, las arquitecturas y estrategias se definen directamente como matriz completa, y el notebook de resultados carga solamente `*_full_summary.json`.

Estado actual:

- `02_cxr_classification.ipynb`: matriz CXR full directa.
- `03_ct_classification.ipynb`: matriz CT full directa.
- `04_classification_results.ipynb`: consolidacion solo de artefactos full.
- `scripts/run_phase1_full.py`: script de ejecucion full para fase 1.

#### 5. `05_segmentation.ipynb` no reproduce toda la matriz final de segmentacion

El notebook `05_segmentation.ipynb` aparece centrado en CT y Attention U-Net, mientras que los resultados finales incluyen tambien:

- CXR U-Net;
- CXR Attention U-Net;
- CT U-Net;
- CT Attention U-Net.

Esto no invalida los resultados, porque los artefactos existen, pero si afecta a reproducibilidad narrativa.

Correccion recomendada:

- dejar `05_segmentation.ipynb` como notebook general de baselines CXR/CT;
- mover las variantes CT a `05b`-`05j`, como ya se ha hecho;
- o aclarar que `05` fue reutilizado como punto de continuacion CT.

#### 6. `05i_ct_negative_slice_training.ipynb` no demuestra por si solo reduccion de falsos positivos

El experimento con slices negativos es buena idea, pero si validacion/test siguen siendo positivos, entonces mide si anadir negativos mejora Dice en slices positivos. No mide de forma directa si el modelo reduce falsos positivos en slices sin lesion.

Conclusion valida:

> Se exploro la inclusion de contexto negativo durante entrenamiento.

Conclusion no demostrada:

> El modelo reduce falsos positivos en slices negativos.

Para demostrarlo haria falta un test con slices negativos y metricas como false positive rate por slice negativo.

#### 7. Grad-CAM tiene muestra pequena

El notebook `06_explainability.ipynb` usa una muestra reducida por clase y casos incorrectos limitados. Esto es razonable por coste computacional y como analisis cualitativo, pero no permite una conclusion poblacional fuerte.

Debe redactarse como:

> auditoria cualitativa y cuantitativa exploratoria de saliencia.

No como:

> validacion estadistica completa de interpretabilidad.

### Prioridad baja

#### 9. Nombre `bf32`

En el mejor modelo CT, el nombre contiene `bf32`. En el contexto del notebook significa `base_features=32`, no `bfloat32`.

Para evitar ambiguedad en la memoria, escribir:

> variante de mayor capacidad con 32 filtros base (`base_features=32`).

#### 10. 2.5D debe presentarse como exploracion, no como mejora

El experimento 2.5D se ejecuto, pero no supero al mejor 2D de mayor capacidad. Es cientificamente valioso como resultado negativo.

Redaccion recomendada:

> Se evaluo una variante 2.5D usando contexto de slices vecinos, pero en esta configuracion no mejoro el mejor modelo 2D. Esto sugiere que el beneficio del contexto inter-slice depende de la configuracion, la calidad de las anotaciones y el coste computacional.

## Revision notebook por notebook

| Notebook | Estado metodologico | Observacion |
|---|---|---|
| `00_eda.ipynb` | Correcto | Justifica distribuciones, desbalanceo, fusion CT-3/CT-4 y disponibilidad de mascaras. |
| `02_cxr_classification.ipynb` | Correcto | Configurado como matriz CXR full. |
| `03_ct_classification.ipynb` | Correcto | Buen punto: split por `study_id` y control de artefactos existentes. |
| `04_classification_results.ipynb` | Correcto | Carga solo summaries full para tablas finales. |
| `05_segmentation.ipynb` | Correcto con aclaracion | Conviene separarlo como baseline general o aclarar que se centro en CT. |
| `05b_ct_tversky_variant.ipynb` | Correcto | Variante ligera justificada por desbalanceo pixel a pixel. |
| `05c_ct_patch_training.ipynb` | Correcto | Resultado negativo util: patch pequeno no mejora. |
| `05d_ct_mixed_context_training.ipynb` | Correcto | Mejora clara al combinar contexto y parches. |
| `05e_ct_postprocessing_analysis.ipynb` | Exploratorio | No usar como seleccion final si se hizo sobre test. |
| `05f_ct_25d_context_training.ipynb` | Correcto | Exploracion 2.5D defendible, aunque no sea el mejor. |
| `05g_ct_mixed_context_ablation.ipynb` | Correcto | Ablacion importante para justificar patch size, proporcion y entrenamiento. |
| `05h_ct_mixed_ensemble.ipynb` | Correcto | Buen uso de validacion para seleccionar pesos/umbral antes de test. |
| `05i_ct_negative_slice_training.ipynb` | Correcto con limitacion | No demuestra FPR en negativos si test no incluye negativos. |
| `05j_ct_high_capacity_refinement.ipynb` | Correcto | Mejor resultado CT; justificar como aumento de capacidad y entrenamiento mas largo. |
| `05k_ct_segmentation_visualization.ipynb` | Correcto | Muy util para defensa cualitativa; separar casos faciles, medios y fallidos. |
| `06_explainability.ipynb` | Correcto como XAI exploratoria | Grad-CAM suficiente si se declara como interpretabilidad post-hoc limitada. |
| `07_final_analysis.ipynb` | Correcto | Usa resultados full y resume RQs; matizar independencia por slice en CT. |
| `08_calibration_analysis.ipynb` | Correcto | Aporta robustez al estudio de clasificacion. |
| `09_ct_study_level_analysis.ipynb` | Correcto | Complementa CT por slice con evaluacion por estudio, mas coherente con las etiquetas de MosMedData. |

## Que se puede defender en la memoria

### Clasificacion CXR

Se puede defender que los modelos CXR alcanzan rendimiento alto en test interno, especialmente DenseNet-121 con weighted cross-entropy. La comparacion entre ResNet-50, DenseNet-121 y EfficientNet-B0 esta bien justificada por transfer learning y por uso de metricas multiclase.

Limite:

> No es validacion clinica externa; el dataset puede contener sesgos de fuente o adquisicion.

### Clasificacion CT

Se puede defender que CT es mas dificil que CXR. El rendimiento menor es coherente con:

- etiquetas ordinales de severidad;
- variabilidad entre slices;
- desbalanceo;
- dependencia entre slices del mismo estudio;
- ausencia de informacion volumetrica completa en modelos 2D.

Esto no es un fracaso del TFM: es una conclusion cientifica importante.

Ademas, la evaluacion por estudio aporta una lectura mas defendible porque la etiqueta CT pertenece al volumen completo. En los resultados actuales, agregar probabilidades por `study_id` mejora el mejor F1-macro CT desde aproximadamente 0.417 por slice hasta 0.516 por estudio, lo que apoya que parte de la dificultad venia de evaluar cortes individuales con etiquetas heredadas del estudio.

### Segmentacion CXR

Se puede defender que U-Net y Attention U-Net segmentan pulmon con rendimiento muy alto. Pero debe quedar claro que:

- la mascara CXR es pulmonar;
- no es mascara de lesion COVID;
- sirve como referencia anatomica para plausibilidad de Grad-CAM, no como verdad de patologia.

### Segmentacion CT

Se puede defender una mejora experimental progresiva:

1. baseline CT aproximadamente Dice 0.50;
2. mixed context mejora hasta aproximadamente Dice 0.52;
3. configuracion mixed30 patch192 mejora hasta aproximadamente Dice 0.53;
4. aumento de capacidad y entrenamiento mas largo llega a aproximadamente Dice 0.56;
5. ensemble mejora frente a algunos modelos, pero no supera la mejor variante individual.

Narrativa fuerte:

> En CT, la mejora no vino solo de cambiar hiperparametros, sino de adaptar el muestreo y el contexto a lesiones pequenas y desbalanceadas.

### Explicabilidad

Se puede defender Grad-CAM como auditoria visual y cuantitativa limitada. En CXR, sirve para comprobar si la activacion cae dentro del pulmon. En CT, la baja superposicion Grad-CAM-lesion sugiere que la clasificacion no se explica de forma localizada por la mascara de infeccion disponible.

Esta conclusion es buena porque evita una lectura exagerada de la IA explicable.

### Calibracion

Se puede defender como fase adicional de fiabilidad. Mide si las probabilidades del modelo son coherentes con su accuracy observada y permite estudiar errores de alta confianza.

## Que no se debe afirmar

Evitar estas afirmaciones en la memoria:

- "El sistema esta validado clinicamente".
- "Grad-CAM demuestra causalmente que el modelo mira la lesion".
- "CXR segmenta lesion COVID"; en CXR segmenta pulmon.
- "El postprocesado mejora el modelo final" si la seleccion se hizo sobre test.
- "El entrenamiento con slices negativos reduce falsos positivos" si no se evalua sobre slices negativos.
- "El 2.5D mejora CT"; en los resultados actuales no mejora el mejor modelo 2D.
- "Las pruebas estadisticas CT son completamente independientes por slice"; son aproximadas si se calculan por slice, aunque ahora se dispone de una evaluacion complementaria por estudio.

## Recomendaciones antes de cerrar la memoria

1. Mantener siempre resultados `full` en tablas principales.
2. Declarar que CT clasificacion es por slice con split agrupado por estudio y anadir la evaluacion complementaria por estudio.
3. Declarar que CT segmentacion se evalua principalmente sobre slices positivos.
4. Usar `05e` solo como analisis exploratorio o repetirlo sobre validacion.
5. Aclarar que `bf32` significa `base_features=32`.
6. Presentar Grad-CAM como XAI cualitativa/exploratoria.
7. Incluir resultados negativos como parte de las ablationes.
8. En la discusion, anadir limitaciones: sin validacion externa, dependencia por slices, posible sesgo de fuente en CXR, muestra limitada en XAI.
9. Si queda tiempo adicional, priorizar un pequeno analisis de slices negativos para segmentacion CT; la tabla por estudio de clasificacion CT ya esta generada.

## Conclusiones de revision

El trabajo tiene una base experimental razonable y defendible. No se aprecia que se hayan hecho experimentos sin sentido. Lo mas importante ahora es redactar con precision:

- que se comparo;
- sobre que unidad se evaluo;
- que se optimizo en validacion;
- que se miro de forma exploratoria;
- que limitaciones quedan.

Si se corrigen o matizan los puntos anteriores, el TFM no solo muestra resultados, sino tambien criterio cientifico: reconoce donde el modelo funciona, donde falla y que decisiones metodologicas explican cada resultado.
