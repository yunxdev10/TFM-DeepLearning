# Estructura recomendada para la seccion de metodologia

Fecha: 2026-05-31

## Objetivo de la metodologia

La metodologia debe explicar de forma reproducible que se hizo en el TFM. No debe repetir todo el estado del arte, sino describir el protocolo experimental: datos, preprocesamiento, particiones, modelos, entrenamiento, metricas, comparaciones y criterios de seleccion.

La estructura recomendada tiene 9 partes principales.

## 3.1. Diseno general del estudio

Funcion: abrir la metodologia con una vista global del pipeline.

Contenido:

- El TFM compara dos modalidades: CXR y CT.
- Se realizan tareas de clasificacion, segmentacion, explicabilidad y calibracion.
- CXR y CT no son equivalentes clinicamente:
  - CXR: clases diagnosticas/radiologicas.
  - CT: severidad radiologica.
- Se mantiene separacion entre entrenamiento, validacion y test.
- Los modelos se seleccionan o ajustan usando validacion; test se reserva para evaluacion final.

Texto clave:

> El estudio se disena como un pipeline experimental comparativo sobre dos modalidades de imagen toracica. La comparacion entre CXR y CT se interpreta de forma metodologica y no como equivalencia clinica directa, ya que las etiquetas, el tipo de imagen y la disponibilidad de mascaras difieren entre ambas modalidades.

## 3.2. Conjuntos de datos

Funcion: describir con precision las fuentes de datos.

Subpartes recomendadas:

### 3.2.1. Dataset CXR

Contenido:

- COVID-19 Radiography Database.
- Clases: COVID-19, Lung Opacity, Normal, Viral Pneumonia.
- Imagenes 2D de radiografia de torax.
- Mascaras pulmonares disponibles para segmentacion pulmonar.
- Aclarar que la mascara CXR no es mascara de lesion COVID.

### 3.2.2. Dataset CT

Contenido:

- MosMedData.
- Estudios CT toracicos.
- Clases CT-0, CT-1, CT-2, CT-3/CT-4 o CT-3+ si se agruparon.
- Subconjunto con mascaras de infeccion/lesion.
- CT es volumetrico, pero el pipeline principal trabaja con slices 2D.
- Separacion por estudio para evitar fuga entre slices del mismo paciente/estudio.

### 3.2.3. Diferencias entre CXR y CT

Contenido:

- Diferente tipo de imagen.
- Diferente significado de etiquetas.
- Diferente disponibilidad y significado de mascaras.
- Diferente dificultad experimental.

## 3.3. Preprocesamiento y particiones

Funcion: explicar como se prepararon los datos antes de entrenar.

Contenido:

- Redimensionado de imagenes.
- Adaptacion de canales:
  - CXR normalmente a 3 canales para modelos preentrenados.
  - CT 2D como slice individual.
  - CT 2.5D como tres canales: slice anterior, actual y posterior.
- Normalizacion.
- Extraccion o seleccion de slices CT.
- Conversion de mascaras a formato binario en segmentacion.
- Particiones train/validation/test.
- Control de fuga de informacion:
  - especialmente en CT por `study_id`.
- Augmentacion solo en entrenamiento.

Importante:

> Validacion y test no deben recibir transformaciones aleatorias. Solo preprocesamiento determinista.

## 3.4. Clasificacion

Funcion: documentar los modelos de clasificacion y como se entrenaron.

Subpartes recomendadas:

### 3.4.1. Arquitecturas

Modelos usados:

- ResNet-50.
- DenseNet-121.
- EfficientNet-B0.

Contenido:

- Modelos preentrenados.
- Sustitucion de la cabeza clasificadora.
- Salida multiclase con softmax en evaluacion.
- Comparacion bajo el mismo protocolo.

### 3.4.2. Transfer learning y fine-tuning

Contenido:

- Uso de pesos preentrenados.
- Adaptacion al numero de clases CXR/CT.
- Entrenamiento con validacion.
- Early stopping si se uso.
- Guardado de predicciones y artefactos.

### 3.4.3. Estrategias de desbalanceo en clasificacion

Estrategias reales:

- Baseline.
- Weighted cross-entropy.
- Focal loss.
- Oversampling / muestreo ponderado.
- Augmentacion como apoyo/regularizacion, no como nueva fuente clinica.

## 3.5. Segmentacion

Funcion: explicar las tareas de segmentacion y las variantes.

Subpartes recomendadas:

### 3.5.1. Tareas de segmentacion

Contenido:

- CXR: segmentacion pulmonar.
- CT: segmentacion de lesion o infeccion.
- Aclarar que no es lo mismo segmentar pulmon que segmentar lesion.

### 3.5.2. Arquitecturas

Arquitecturas usadas:

- U-Net.
- Attention U-Net.

No incluir como usado:

- ResUNet.
- 3D U-Net.

Se pueden citar en estado del arte, pero no en metodologia como modelos implementados.

### 3.5.3. Perdidas y entrenamiento

Contenido:

- Dice + BCE baseline.
- Weighted Tversky + BCE para CT.
- `pos_weight` para aumentar la importancia de pixeles positivos.
- Dice e IoU como metricas principales.
- Pixel accuracy como metrica secundaria, especialmente poco fiable en CT por dominio del fondo.

### 3.5.4. Variantes CT y ablaciones

Aqui esta una parte muy importante porque muestra experimentacion real.

Variantes a documentar:

- Baseline U-Net y Attention U-Net.
- Tversky ponderada.
- Entrenamiento por parches.
- Positive crop sampling.
- Mixed context training.
- Ajuste de threshold en validacion.
- Postprocesado ligero / componentes conectados si se reporta.
- 2.5D CT como exploracion.
- Ensemble por promedio de probabilidades.
- Aumento de capacidad con `base_features=32`.

Lectura metodologica:

> Las variantes CT se plantean como un estudio de ablacion progresivo para analizar que decisiones ayudan a mejorar una tarea con lesion pequena y fuerte desbalance pixel a pixel.

## 3.6. Explicabilidad mediante Grad-CAM

Funcion: documentar solo lo que se implemento.

Contenido:

- Grad-CAM como metodo XAI principal.
- Modelos clasificadores seleccionados.
- Generacion de mapas de saliencia.
- Binarizacion o seleccion de regiones de mayor activacion.
- Comparacion con mascaras disponibles.

Diferencia por modalidad:

- CXR: Grad-CAM vs mascara pulmonar, mide plausibilidad anatomica.
- CT: Grad-CAM vs mascara de lesion/infeccion, mide alineacion con region patologica anotada.

No poner como metodologia implementada:

- SHAP.
- LIME.

Pueden mencionarse en estado del arte o trabajo futuro, no en metodologia/resultados.

## 3.7. Calibracion probabilistica

Funcion: incluir el notebook 08 si se decide mantenerlo como fase final.

Contenido:

- Analisis sin reentrenar modelos.
- Uso de predicciones guardadas.
- Metricas:
  - confianza maxima media,
  - ECE,
  - MCE,
  - Brier score multiclase,
  - negative log-likelihood,
  - errores de alta confianza.
- Diagramas de fiabilidad e histogramas de confianza.

Si se quiere reducir alcance:

> Presentarlo como analisis complementario, no como eje principal del TFM.

## 3.8. Metricas de evaluacion

Funcion: reunir las metricas usadas y justificar por que se usan.

Clasificacion:

- Accuracy.
- Precision.
- Recall/sensibilidad.
- F1-score por clase.
- F1-macro.
- F1-weighted.
- AUC macro one-vs-rest.
- Matriz de confusion.
- Intervalos de confianza por bootstrap si se reportan.
- McNemar para comparacion entre mejores modelos si se mantiene.

Segmentacion:

- Dice.
- IoU/Jaccard.
- Pixel accuracy como secundaria.
- Analisis cualitativo de ejemplos.

Explicabilidad:

- IoU saliencia-mascara.
- Ratio de saliencia dentro de mascara.
- Pico de activacion dentro de mascara.

Calibracion:

- ECE.
- MCE.
- Brier score.
- NLL.
- Errores de alta confianza.

## 3.9. Diseno experimental y reproducibilidad

Funcion: cerrar metodologia explicando como se comparan experimentos.

Contenido:

- Los experimentos documentados en la memoria corresponden a ejecuciones `full`.
- Seleccion de hiperparametros y thresholds en validacion.
- Evaluacion final en test.
- Guardado de artefactos:
  - summaries JSON,
  - CSV de predicciones,
  - matrices de confusion,
  - figuras,
  - metricas de segmentacion,
  - mapas Grad-CAM,
  - metricas de calibracion.
- Trazabilidad de notebooks y scripts.

Texto clave:

> Los resultados reportados en este TFM corresponden a ejecuciones full y a evaluaciones sobre el conjunto de test reservado. La seleccion de configuraciones y umbrales se realiza mediante validacion, evitando ajustar decisiones metodologicas directamente sobre test.

## Orden recomendado del capitulo

```text
3. Metodologia
3.1. Diseno general del estudio
3.2. Conjuntos de datos
  3.2.1. COVID-19 Radiography Database
  3.2.2. MosMedData
  3.2.3. Diferencias entre modalidades y etiquetas
3.3. Preprocesamiento, particiones y control de fuga
3.4. Clasificacion
  3.4.1. Arquitecturas
  3.4.2. Transfer learning y fine-tuning
  3.4.3. Estrategias de desbalanceo
3.5. Segmentacion
  3.5.1. Tareas CXR y CT
  3.5.2. U-Net y Attention U-Net
  3.5.3. Perdidas y metricas de entrenamiento
  3.5.4. Variantes CT y ablaciones
3.6. Explicabilidad mediante Grad-CAM
3.7. Calibracion probabilistica
3.8. Metricas de evaluacion
3.9. Diseno experimental y reproducibilidad
```

## Que no debe faltar

- La metodologia debe dejar claro que CXR y CT no son tareas clinicamente equivalentes.
- CXR segmenta pulmones, no lesiones COVID.
- CT segmenta lesion/infeccion.
- Grad-CAM no es una mascara ni prueba causal.
- 2.5D fue exploracion, no mejor modelo final.
- Los thresholds se seleccionan en validacion cuando forman parte del metodo.
- La memoria debe reportar solo resultados `full`.
- La metrica principal de segmentacion CT es Dice/IoU, no pixel accuracy.
