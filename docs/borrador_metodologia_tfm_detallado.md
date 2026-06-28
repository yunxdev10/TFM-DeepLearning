# Borrador detallado de metodologia del TFM

Fecha: 2026-06-05

Este documento amplia el borrador de metodologia con mas detalle explicativo. La idea es que la metodologia no sea solo una lista de herramientas, sino una explicacion clara de como se construyo el pipeline, por que se tomaron ciertas decisiones y como debe interpretarse cada parte.

La redaccion esta pensada para adaptarse a la memoria en LaTeX. Se mantiene una advertencia importante: aqui se documenta lo que realmente se hizo. Por tanto, no se presenta SHAP/LIME como metodologia implementada, no se usa ResUNet, no se entrena 3D U-Net y el enfoque 2.5D se describe como exploracion experimental en CT.

## Idea general de la metodologia

La metodologia debe responder a estas preguntas:

- Que datos se usaron.
- Como se dividieron los datos.
- Como se prepararon las imagenes.
- Que modelos se entrenaron.
- Como se trato el desbalanceo.
- Como se evaluaron los resultados.
- Como se interpreto la explicabilidad.
- Como se aseguro que las comparaciones fueran reproducibles.

Una forma sencilla de explicar el estudio es:

> Este TFM construye un pipeline experimental de aprendizaje profundo sobre dos tipos de imagen toracica, CXR y CT. Primero se preparan los datos y se separan en entrenamiento, validacion y prueba. Despues se entrenan clasificadores CNN para predecir clases diagnosticas o grados de severidad. En paralelo, se entrenan modelos de segmentacion para delimitar pulmones en CXR y lesiones/infeccion en CT. Finalmente, se aplica Grad-CAM para inspeccionar visualmente las decisiones de los clasificadores y se analiza la calibracion para estudiar si la confianza de los modelos es fiable.

## 3. Metodologia

```latex
\section{Metodologia}
```

La metodologia describe el protocolo experimental seguido para evaluar modelos de aprendizaje profundo en imagen toracica relacionada con COVID-19. El trabajo se organiza alrededor de dos modalidades de imagen: radiografia de torax (CXR) y tomografia computarizada (CT). Ambas modalidades se estudian con un mismo enfoque general, pero no se consideran equivalentes desde el punto de vista clinico.

En CXR, el problema principal de clasificacion consiste en distinguir entre cuatro categorias radiologicas: COVID-19, Lung Opacity, Normal y Viral Pneumonia. En CT, en cambio, el problema se plantea como clasificacion de severidad radiologica usando categorias derivadas de MosMedData. Por ello, comparar CXR y CT no significa comparar exactamente el mismo diagnostico en dos formatos distintos, sino analizar como se comportan modelos de deep learning bajo dos modalidades medicas relacionadas pero diferentes.

Ademas de la clasificacion, el TFM incorpora segmentacion, explicabilidad y calibracion. La segmentacion permite estudiar regiones espaciales: en CXR se segmentan pulmones, mientras que en CT se segmentan regiones de infeccion o lesion. La explicabilidad se estudia mediante Grad-CAM, generando mapas de saliencia que indican que zonas influyen en la prediccion del clasificador. La calibracion analiza si la confianza probabilistica del modelo se corresponde con su tasa real de acierto.

## 3.1. Diseno general del estudio

```latex
\subsection{Diseno general del estudio}
```

El estudio se estructura en cinco bloques:

1. Preparacion de datos.
2. Clasificacion.
3. Segmentacion.
4. Explicabilidad mediante Grad-CAM.
5. Calibracion probabilistica.

Cada bloque cumple una funcion distinta. La preparacion de datos asegura que las imagenes se encuentran en un formato adecuado y que las particiones de entrenamiento, validacion y prueba son reproducibles. La clasificacion evalua la capacidad de distintas arquitecturas para asignar una imagen a una clase. La segmentacion evalua la capacidad de delimitar regiones anatomicas o patologicas. La explicabilidad analiza si las zonas que influyen en la clasificacion son espacialmente plausibles. La calibracion estudia si la confianza del modelo es razonable.

La separacion entre entrenamiento, validacion y prueba se mantiene durante todo el trabajo. El conjunto de entrenamiento se utiliza para ajustar los pesos de los modelos. El conjunto de validacion se emplea para tomar decisiones durante el desarrollo, como early stopping, seleccion de umbral o configuracion de ensemble. El conjunto de prueba se reserva para la evaluacion final. Esta separacion es importante porque evita ajustar decisiones directamente sobre el test y reduce el riesgo de obtener resultados artificialmente optimistas.

Texto listo para memoria:

```latex
El pipeline experimental se organiza en cinco bloques principales: preparacion de datos, clasificacion, segmentacion, explicabilidad y calibracion. En primer lugar, se preparan los datasets CXR y CT y se generan particiones reproducibles de entrenamiento, validacion y prueba. En segundo lugar, se entrenan clasificadores CNN basados en transfer learning. En tercer lugar, se entrenan modelos de segmentacion sobre las mascaras disponibles. En cuarto lugar, se aplican mapas Grad-CAM para estudiar la saliencia visual de los clasificadores. Finalmente, se realiza un analisis complementario de calibracion probabilistica a partir de las predicciones guardadas.

La comparacion entre CXR y CT se interpreta de forma metodologica, no como equivalencia clinica directa. Las etiquetas, la naturaleza de la imagen y las mascaras disponibles difieren entre ambas modalidades. Por tanto, los resultados se analizan teniendo en cuenta estas diferencias.
```

## 3.2. Conjuntos de datos

```latex
\subsection{Conjuntos de datos}
```

El TFM utiliza dos datasets principales:

| Modalidad | Dataset | Uso principal |
|---|---|---|
| CXR | COVID-19 Radiography Database | Clasificacion CXR y segmentacion pulmonar |
| CT | MosMedData | Clasificacion de severidad CT y segmentacion de infeccion/lesion |

### 3.2.1. Dataset CXR

```latex
\subsubsection{COVID-19 Radiography Database}
```

El dataset CXR utilizado es COVID-19 Radiography Database. Contiene radiografias de torax en cuatro clases:

- COVID-19.
- Lung Opacity.
- Normal.
- Viral Pneumonia.

Estas clases se usan para clasificacion multiclase. Es decir, para cada imagen el modelo debe elegir una de las cuatro categorias posibles. En el codigo, las imagenes se leen desde las carpetas de cada clase y se construye un dataframe con la ruta de la imagen y la etiqueta correspondiente.

El dataset tambien contiene mascaras pulmonares. Estas mascaras se utilizan en dos partes del TFM:

- Para entrenar modelos de segmentacion pulmonar en CXR.
- Para comparar si la saliencia Grad-CAM cae dentro del campo pulmonar.

Una aclaracion importante para la memoria:

> La mascara pulmonar no indica lesion COVID. Solo delimita la region anatomica de los pulmones. Por tanto, en CXR no se evalua si Grad-CAM localiza una lesion concreta, sino si la saliencia se encuentra en una zona anatomica razonable.

Split CXR verificado:

| Split | Numero de imagenes |
|---|---:|
| Entrenamiento | 14.815 |
| Validacion | 3.175 |
| Test | 3.175 |

Distribucion de clases en entrenamiento:

| Clase | Imagenes train |
|---|---:|
| COVID | 2.531 |
| Lung Opacity | 4.208 |
| Normal | 7.134 |
| Viral Pneumonia | 942 |

Texto listo para memoria:

```latex
Para la modalidad CXR se utiliza COVID-19 Radiography Database, compuesto por radiografias de torax organizadas en cuatro clases: COVID-19, Lung Opacity, Normal y Viral Pneumonia. El problema se formula como clasificacion multiclase. Adicionalmente, el dataset incluye mascaras pulmonares, que se utilizan para entrenar modelos de segmentacion pulmonar y para evaluar la plausibilidad anatomica de los mapas Grad-CAM.

Las mascaras CXR delimitan el campo pulmonar, pero no representan lesiones COVID-19. Por este motivo, cualquier comparacion entre Grad-CAM y mascara pulmonar debe interpretarse como una medida de plausibilidad anatomica, no como localizacion patologica.
```

### 3.2.2. Dataset CT

```latex
\subsubsection{MosMedData}
```

Para CT se utiliza MosMedData, un dataset de estudios de tomografia computarizada toracica relacionados con COVID-19. A diferencia del dataset CXR, que contiene imagenes 2D individuales, MosMedData contiene estudios volumetricos. Cada estudio CT esta formado por multiples slices o cortes axiales.

En clasificacion, las clases originales son:

- CT-0.
- CT-1.
- CT-2.
- CT-3.
- CT-4.

En el pipeline se agrupan CT-3 y CT-4 en una unica clase CT-3+. Esta agrupacion se realiza porque las clases severas tienen menos ejemplos y separarlas puede producir una evaluacion menos estable. Esta decision debe explicarse como una decision metodologica, no como una nueva etiqueta clinica original.

Para clasificacion CT se extraen slices 2D desde los volumenes. El proceso es:

1. Cargar el volumen NIfTI.
2. Aplicar ventana de Hounsfield en el rango `[-1000, 400]`.
3. Normalizar intensidades a rango de imagen.
4. Seleccionar los cortes centrales entre el 20% y el 80% del volumen.
5. Rotar el slice para orientarlo correctamente.
6. Redimensionar a `256x256`.
7. Guardar cada slice como PNG.

La ventana `[-1000, 400]` se usa porque esta centrada en el tejido pulmonar y permite resaltar estructuras relevantes del torax. De forma sencilla: se recortan intensidades fuera de ese rango para que el modelo vea mejor la informacion pulmonar en lugar de recibir valores extremos poco utiles.

La particion CT se hace por `study_id`, no por slice. Esto es crucial. Si slices del mismo estudio aparecieran en train y test, el modelo podria ver imagenes muy parecidas durante entrenamiento y evaluacion, produciendo fuga de informacion. Por tanto, todos los slices de un mismo estudio se asignan al mismo split.

Split CT de clasificacion verificado:

| Split | Slices | Estudios |
|---|---:|---:|
| Entrenamiento | 19.456 | 777 |
| Validacion | 4.170 | 166 |
| Test | 4.155 | 167 |

Distribucion de clases en entrenamiento CT:

| Clase | Slices train |
|---|---:|
| CT-0 | 4.644 |
| CT-1 | 11.858 |
| CT-2 | 2.151 |
| CT-3+ | 803 |

MosMedData tambien contiene un subconjunto con mascaras de infeccion o lesion. Para segmentacion CT se usan slices con mascara positiva. Esto significa que la tarea se centra en aprender la delimitacion de regiones anotadas como infeccion/lesion.

Split CT de segmentacion verificado:

| Split | Slices | Estudios |
|---|---:|---:|
| Entrenamiento | 508 | 35 |
| Validacion | 86 | 7 |
| Test | 110 | 8 |

Texto listo para memoria:

```latex
Para CT se utiliza MosMedData, formado por estudios volumetricos de tomografia computarizada. En clasificacion, las clases CT-3 y CT-4 se agrupan como CT-3+ para reducir la fragmentacion de las clases severas y mejorar la estabilidad experimental. Los volumenes se convierten en slices 2D aplicando una ventana de Hounsfield $[-1000,400]$, seleccionando la region central del volumen y redimensionando cada corte a $256 \times 256$ pixeles.

La division de datos CT se realiza por identificador de estudio. Esta decision evita que slices del mismo estudio aparezcan simultaneamente en entrenamiento y prueba, lo que podria producir fuga de informacion. Por tanto, el split CT se define a nivel de estudio y despues se propaga a todos los slices asociados.
```

## 3.3. Preprocesamiento y augmentacion

```latex
\subsection{Preprocesamiento y augmentacion}
```

El preprocesamiento no es identico en CXR y CT porque las modalidades tienen formatos distintos.

### CXR

En CXR, las imagenes se cargan como RGB. Aunque muchas radiografias son visualmente en escala de grises, se convierten a tres canales porque ResNet-50, DenseNet-121 y EfficientNet-B0 fueron preentrenadas con imagenes RGB de ImageNet.

Transformaciones en entrenamiento CXR:

- Resize a `224x224`.
- RandomHorizontalFlip.
- RandomRotation de hasta 15 grados.
- RandomAffine con pequenas traslaciones y escala.
- ColorJitter suave en brillo y contraste.
- Conversion a tensor.
- Normalizacion ImageNet.

Transformaciones en validacion/test CXR:

- Resize a `224x224`.
- Conversion a tensor.
- Normalizacion ImageNet.

La diferencia es importante: en entrenamiento se introduce variabilidad para reducir sobreajuste, pero en validacion y test no se alteran aleatoriamente las imagenes. Asi las metricas reflejan el rendimiento sobre datos evaluados de forma estable.

### CT

En CT, los slices se cargan como imagenes de un canal. Para clasificacion se usa normalizacion con media 0,5 y desviacion 0,5.

Transformaciones en entrenamiento CT:

- Resize a `256x256`.
- RandomHorizontalFlip.
- RandomRotation de hasta 5 grados.
- RandomAffine suave.
- Conversion a tensor.
- Normalizacion.

Las transformaciones CT son mas conservadoras que en CXR porque la anatomia volumetrica y la orientacion del cuerpo deben preservarse lo maximo posible.

### Segmentacion

En segmentacion, cualquier transformacion geometrica debe aplicarse de forma sincronizada a imagen y mascara. Por ejemplo, si se rota la imagen pero no la mascara, la mascara deja de corresponder con la region real. Por eso el pipeline usa transformaciones pareadas.

En segmentacion se aplican:

- Resize de imagen y mascara.
- Volteo horizontal conjunto.
- Rotacion conjunta.
- Recortes por parches en variantes CT.
- Recortes positivos en variantes CT.

Texto listo para memoria:

```latex
Las transformaciones aleatorias se aplican exclusivamente durante entrenamiento. En validacion y test se utiliza un preprocesamiento determinista. En segmentacion, las transformaciones geometricas se aplican simultaneamente sobre imagen y mascara para conservar la correspondencia espacial entre ambas.
```

## 3.4. Clasificacion

```latex
\subsection{Clasificacion}
```

La clasificacion evalua la capacidad de distintas CNN para predecir la clase de cada imagen o slice.

### Arquitecturas

Se entrenan tres arquitecturas:

| Arquitectura | Motivo de inclusion |
|---|---|
| ResNet-50 | Baseline profundo con conexiones residuales |
| DenseNet-121 | Reutilizacion densa de caracteristicas |
| EfficientNet-B0 | Arquitectura eficiente en parametros y coste |

Todas se usan con transfer learning. Es decir, parten de pesos preentrenados y se adaptan al nuevo problema. Esta decision es importante porque en imagen medica los datasets anotados suelen ser mas limitados que los datasets generales como ImageNet.

### Adaptacion de la entrada

En CXR:

- Entrada de 3 canales.
- Imagenes RGB.
- Normalizacion ImageNet.

En CT:

- Entrada de 1 canal.
- Imagenes en escala de grises.
- Primera convolucion adaptada a 1 canal.

La adaptacion a 1 canal se hace promediando los pesos preentrenados de los tres canales originales. Explicado facil: el modelo preentrenado espera tres canales, pero CT tiene uno; en vez de empezar desde cero, se promedian los filtros RGB para obtener filtros iniciales compatibles con imagenes de un canal.

### Entrenamiento en dos fases

La clasificacion se entrena con una estrategia de dos fases:

1. Entrenamiento de la cabeza clasificadora.
2. Fine-tuning del modelo completo.

Fase 1:

- Se congela el backbone.
- Solo se entrena la capa final.
- Duracion: 5 epocas.
- Learning rate: `1e-4`.

Fase 2:

- Se descongela todo el modelo.
- Se ajustan tambien las capas convolucionales.
- Duracion: 15 epocas.
- Learning rate: `1e-5`.

Hiperparametros de clasificacion:

| Parametro | Valor |
|---|---:|
| Batch size | 32 |
| Epochs totales | 20 |
| Head epochs | 5 |
| Fine-tuning epochs | 15 |
| Learning rate inicial | `1e-4` |
| Learning rate fine-tuning | `1e-5` |
| Weight decay | `1e-5` |
| Optimizer | AdamW |
| Early stopping patience | 5 |
| Scheduler | ReduceLROnPlateau |
| Seed | 42 |

Texto listo para memoria:

```latex
El entrenamiento de clasificacion se realiza en dos fases. Primero se congela el backbone preentrenado y se entrena la cabeza clasificadora durante 5 epocas. Posteriormente se descongela el modelo completo y se realiza fine-tuning durante 15 epocas adicionales con una tasa de aprendizaje menor. Esta estrategia permite adaptar progresivamente las representaciones preentrenadas al dominio medico sin destruir bruscamente los pesos aprendidos.
```

### Estrategias de desbalanceo

El desbalanceo es importante porque algunas clases tienen muchas mas muestras que otras. Si no se controla, el modelo puede aprender a favorecer la clase mayoritaria y obtener buena accuracy global pero mal rendimiento en clases minoritarias.

Se prueban cuatro estrategias:

| Estrategia | Como funciona | Que intenta corregir |
|---|---|---|
| Baseline | CrossEntropyLoss normal | Punto de comparacion sin correccion |
| Weighted CE | Penaliza mas errores en clases minoritarias | Sesgo hacia clases frecuentes |
| Focal loss | Da mas peso a ejemplos dificiles | Dominio de ejemplos faciles |
| Oversampling | Muestra clases minoritarias con mas frecuencia | Poca presencia de clases raras en batches |

Weighted cross-entropy usa pesos inversamente proporcionales a la frecuencia de clase:

```text
peso_clase = total_muestras / (numero_clases * muestras_de_la_clase)
```

Focal loss usa `gamma=2`. Oversampling se implementa con `WeightedRandomSampler`.

Texto listo para memoria:

```latex
Para analizar el efecto del desbalanceo, cada arquitectura se evalua con cuatro estrategias: baseline, weighted cross-entropy, focal loss y oversampling. Weighted cross-entropy modifica la funcion de perdida, focal loss reduce el peso de ejemplos faciles y oversampling modifica la frecuencia con la que las clases aparecen durante el entrenamiento. Estas estrategias se comparan con el baseline para comprobar si realmente mejoran el rendimiento macro y la sensibilidad de clases minoritarias.
```

## 3.5. Segmentacion

```latex
\subsection{Segmentacion}
```

La segmentacion permite pasar de una decision global a una prediccion pixel a pixel. En este TFM hay dos tareas distintas:

| Modalidad | Tarea | Mascara |
|---|---|---|
| CXR | Segmentacion pulmonar | Mascara pulmonar |
| CT | Segmentacion de infeccion/lesion | Mascara patologica |

Esto debe quedar muy claro: en CXR se segmentan pulmones, no lesiones. En CT se segmentan lesiones/infeccion.

### Arquitecturas

Se implementan:

- U-Net.
- Attention U-Net.

No se implementan:

- ResUNet.
- 3D U-Net.

U-Net funciona como encoder-decoder. El encoder reduce resolucion y aprende caracteristicas abstractas; el decoder recupera resolucion para producir una mascara. Las skip connections conectan encoder y decoder para no perder detalle espacial.

Attention U-Net mantiene la misma idea, pero filtra las skip connections mediante mecanismos de atencion. Explicado facil: no pasa toda la informacion del encoder al decoder, sino que aprende a destacar partes mas relevantes.

### Salida del modelo

La salida es un mapa de logits de un canal. Despues:

1. Se aplica sigmoid.
2. Se obtiene una probabilidad por pixel.
3. Se aplica un threshold.
4. Se genera una mascara binaria.

### Perdidas

Se usan dos familias principales:

| Perdida | Uso |
|---|---|
| Dice + BCE | Baseline de segmentacion |
| Weighted Tversky + BCE | Variantes CT con desbalance fuerte |

Dice ayuda a optimizar solapamiento entre prediccion y mascara. BCE evalua pixel a pixel. Tversky permite controlar falsos positivos y falsos negativos.

En CT se usa:

- `tversky_alpha = 0.3`
- `tversky_beta = 0.7`

Esto penaliza mas los falsos negativos. En palabras sencillas: se intenta evitar que el modelo ignore lesiones pequenas.

### Hiperparametros base de segmentacion

| Parametro | Valor |
|---|---:|
| Image size CXR | `224x224` |
| Image size CT | `256x256` |
| Batch size base | 8 |
| Epochs base | 30 |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Early stopping patience base | 6 |
| Base features base | 32 |
| Optimizer | AdamW |

### Variantes CT

CT fue la parte mas experimental de segmentacion. La razon es que las lesiones ocupan pocos pixeles y eso hace que el problema sea mucho mas dificil que segmentar pulmones en CXR.

Resumen de variantes:

| Variante | Idea | Interpretacion |
|---|---|---|
| U-Net baseline | Modelo base sin atencion | Comparador inicial |
| Attention U-Net baseline | Atencion en skip connections | Mejor punto de partida CT |
| Tversky ponderada | Penalizar mas lesion | Atacar desbalance pixel a pixel |
| Patch training | Entrenar con recortes | Ver mas regiones informativas |
| Positive crop sampling | Preferir parches con lesion | Aumentar frecuencia de pixeles positivos |
| Mixed context | Combinar parches y contexto completo | Equilibrar detalle local y anatomia global |
| Threshold tuning | Seleccionar umbral en validacion | Evitar usar 0.5 por defecto |
| 2.5D | Usar slices vecinos como canales | Explorar contexto volumetrico local |
| Ensemble | Promediar probabilidades de modelos | Aprovechar complementariedad |
| Base features 32 | Aumentar capacidad | Mejorar representacion del modelo |

La variante final individual mas fuerte fue:

`ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`

Interpretacion del nombre:

- `mixed30`: 30% de entrenamiento con parches/contexto local.
- `patch192`: parches de `192x192`.
- `pos70`: 70% de probabilidad de centrar el parche en region positiva.
- `tversky_pos10`: Tversky+BCE con `pos_weight=10`.
- `bf32`: `base_features=32`, no bfloat32.
- `thr095`: busqueda de umbral hasta 0.95.

El ensemble combina dos modelos:

- `mixed50_patch160_pos80_tversky_pos20`
- `mixed30_patch192_pos70_tversky_pos10`

El peso y el threshold del ensemble se seleccionan en validacion, no en test.

Texto listo para memoria:

```latex
La segmentacion CT se aborda como un estudio de ablacion progresivo. Partiendo de U-Net y Attention U-Net, se introducen perdidas ponderadas, entrenamiento por parches, muestreo positivo, contexto mixto, seleccion de umbral en validacion, una variante 2.5D y un ensemble de probabilidades. El objetivo no es solo obtener el mejor valor final, sino analizar que decisiones ayudan en una tarea caracterizada por lesiones pequenas y fuerte desbalance pixel a pixel.
```

## 3.6. Explicabilidad mediante Grad-CAM

```latex
\subsection{Explicabilidad mediante Grad-CAM}
```

La explicabilidad se realiza con Grad-CAM. Grad-CAM genera un mapa de calor que indica que zonas de la imagen influyen mas en una clase predicha.

Explicado de forma simple:

> El modelo toma una decision de clasificacion. Grad-CAM mira las activaciones internas de la CNN y los gradientes de la clase predicha para estimar que regiones de la imagen han contribuido mas a esa decision.

Capas objetivo:

| Arquitectura | Capa Grad-CAM |
|---|---|
| ResNet-50 | Ultimo bloque de `layer4` |
| DenseNet-121 | `denseblock4` |
| EfficientNet-B0 | Ultimo bloque de `features` |

Proceso:

1. Cargar modelo entrenado.
2. Seleccionar capa objetivo.
3. Calcular Grad-CAM para una clase objetivo.
4. Normalizar mapa a `[0,1]`.
5. Redimensionar mapa al tamano de entrada.
6. Binarizar saliencia usando cuantil `0.80`.
7. Comparar con mascara disponible.

Metricas XAI:

| Metrica | Que mide |
|---|---|
| IoU saliencia-mascara | Solapamiento entre zona saliente y mascara |
| Ratio dentro de mascara | Proporcion de saliencia que cae dentro de la mascara |
| Pico dentro de mascara | Si el maximo de saliencia cae dentro de la mascara |

Diferencia por modalidad:

- En CXR, se compara Grad-CAM con mascara pulmonar.
- En CT, se compara Grad-CAM con mascara de infeccion/lesion.

Interpretacion:

- CXR: mide plausibilidad anatomica.
- CT: mide alineacion con region patologica anotada.

Limitacion:

> Grad-CAM no es una segmentacion. No demuestra causalidad. Solo ayuda a inspeccionar visualmente si la decision del modelo parece apoyarse en zonas razonables.

Texto listo para memoria:

```latex
En este TFM, Grad-CAM se utiliza como tecnica principal de explicabilidad visual. Los mapas generados se comparan con las mascaras disponibles mediante IoU, proporcion de saliencia dentro de la mascara y localizacion del pico de activacion. En CXR, esta comparacion evalua plausibilidad anatomica porque la mascara es pulmonar. En CT, evalua alineacion con lesion porque la mascara corresponde a infeccion anotada.
```

## 3.7. Calibracion probabilistica

```latex
\subsection{Calibracion probabilistica}
```

La calibracion responde a una pregunta distinta de la accuracy:

> Cuando el modelo dice que tiene 90% de confianza, acierta aproximadamente el 90% de las veces?

Un modelo puede tener buena accuracy y estar mal calibrado. Esto es importante en medicina porque un error con mucha confianza puede ser mas preocupante que un error con baja confianza.

La calibracion se calcula sin reentrenar modelos. Se usan los CSV de predicciones guardados.

Metricas:

| Metrica | Explicacion sencilla |
|---|---|
| Mean confidence | Confianza media del modelo |
| ECE | Diferencia media entre confianza y accuracy por bins |
| MCE | Mayor error de calibracion entre bins |
| Brier score | Error cuadratico entre probabilidades y etiqueta real |
| NLL | Penaliza probabilidades bajas asignadas a la clase correcta |
| Errores de alta confianza | Errores con confianza >= 0.90 |

Tambien se generan:

- Diagramas de fiabilidad.
- Histogramas de confianza.
- Tabla de errores de alta confianza.

Texto listo para memoria:

```latex
La calibracion probabilistica se analiza como fase complementaria a partir de las predicciones guardadas. Se calculan ECE, MCE, Brier score, negative log-likelihood y errores de alta confianza. Este analisis permite estudiar no solo si el modelo acierta, sino tambien si la confianza asignada a sus predicciones es coherente con su rendimiento real.
```

## 3.8. Metricas de evaluacion

```latex
\subsection{Metricas de evaluacion}
```

### Clasificacion

Metricas usadas:

- Accuracy.
- Precision.
- Recall.
- F1-score por clase.
- F1-macro.
- F1-weighted.
- AUC macro one-vs-rest.
- Matriz de confusion.
- Intervalos de confianza bootstrap.
- McNemar entre mejores modelos.

Explicacion sencilla:

- Accuracy dice cuantos casos se aciertan en total.
- Recall dice cuantos casos reales de una clase se detectan.
- Precision dice cuantas predicciones de una clase son correctas.
- F1 combina precision y recall.
- F1-macro da el mismo peso a todas las clases.
- F1-weighted pondera por frecuencia de clase.
- AUC mide separacion probabilistica entre clases.
- Matriz de confusion muestra donde se equivoca el modelo.

Por que F1-macro es importante:

> En datasets desbalanceados, accuracy puede ser alta aunque el modelo falle clases pequenas. F1-macro evita que una clase mayoritaria domine la evaluacion.

### Segmentacion

Metricas:

- Dice.
- IoU/Jaccard.
- Pixel accuracy.

Interpretacion:

- Dice e IoU miden solapamiento.
- IoU suele ser mas estricto.
- Pixel accuracy puede ser enganosa en CT porque casi todo es fondo.

Texto recomendado:

```latex
En segmentacion se priorizan Dice e IoU, ya que cuantifican el solapamiento entre la mascara predicha y la mascara real. Pixel accuracy se reporta como metrica secundaria, pero no se utiliza como criterio principal en CT debido al fuerte predominio del fondo.
```

### Explicabilidad

Metricas:

- IoU saliencia-mascara.
- Ratio de saliencia dentro de mascara.
- Pico de saliencia dentro de mascara.

Estas metricas no validan clinicamente el modelo. Solo permiten cuantificar de forma reproducible si la saliencia se concentra en una region esperada.

### Calibracion

Metricas:

- ECE.
- MCE.
- Brier score.
- NLL.
- Errores de alta confianza.

## 3.9. Diseno experimental y reproducibilidad

```latex
\subsection{Diseno experimental y reproducibilidad}
```

El proyecto guarda todos los artefactos necesarios para revisar los resultados:

- Checkpoints de modelos.
- Historiales de entrenamiento.
- Summaries JSON.
- Predicciones CSV.
- Classification reports.
- Matrices de confusion.
- Metricas de segmentacion.
- Figuras cualitativas.
- Mapas Grad-CAM.
- Metricas de calibracion.

La configuracion central define:

| Elemento | Valor |
|---|---|
| Semilla | 42 |
| CXR image size | `224x224` |
| CT image size | `256x256` |
| Clasificacion batch size | 32 |
| Segmentacion batch size base | 8 |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |

Los resultados finales se consolidan mediante notebooks y scripts:

- `04_classification_results.ipynb`
- `05_segmentation.ipynb` y variantes `05b`-`05k`
- `06_explainability.ipynb`
- `07_final_analysis.ipynb`
- `08_calibration_analysis.ipynb`
- `scripts/build_final_analysis.py`
- `scripts/build_calibration_analysis.py`

Texto listo para memoria:

```latex
La reproducibilidad se garantiza mediante una configuracion centralizada, semilla fija y guardado sistematico de artefactos. Cada experimento genera checkpoints, historiales, predicciones, metricas y resumenes en formato estructurado. Los resultados finales se consolidan a partir de estos artefactos, manteniendo la trazabilidad entre cada valor reportado y el experimento que lo produjo.
```

## Tabla final de decisiones metodologicas

| Bloque | Decision |
|---|---|
| Modalidades | CXR y CT |
| CXR dataset | COVID-19 Radiography Database |
| CT dataset | MosMedData |
| CXR clases | COVID, Lung Opacity, Normal, Viral Pneumonia |
| CT clases | CT-0, CT-1, CT-2, CT-3+ |
| CXR split | 14.815 / 3.175 / 3.175 |
| CT clasificacion split | 19.456 / 4.170 / 4.155 slices |
| CT clasificacion estudios | 777 / 166 / 167 |
| CT segmentacion split | 508 / 86 / 110 slices |
| CT segmentacion estudios | 35 / 7 / 8 |
| Clasificadores | ResNet-50, DenseNet-121, EfficientNet-B0 |
| Segmentadores | U-Net, Attention U-Net |
| XAI | Grad-CAM |
| Calibracion | ECE, MCE, Brier, NLL |
| CT 2.5D | Exploracion, no enfoque principal |
| CXR segmentacion | Pulmonar |
| CT segmentacion | Lesion/infeccion |

## Errores que hay que evitar en la memoria

- No decir que CXR segmenta lesiones COVID.
- No decir que las mascaras CXR son mascaras patologicas.
- No decir que CT y CXR tienen etiquetas clinicamente equivalentes.
- No decir que SHAP o LIME se implementaron.
- No decir que se uso ResUNet.
- No decir que se entreno 3D U-Net.
- No interpretar Grad-CAM como una mascara.
- No interpretar pixel accuracy CT como metrica principal.
- No presentar 2.5D como mejor modelo final.
- No interpretar `bf32` como bfloat32; significa `base_features=32`.
