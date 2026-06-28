# Borrador de metodologia del TFM

Fecha: 2026-06-05

Este documento contiene un borrador redactado de la seccion de metodologia, basado en los artefactos realmente generados en el proyecto. Esta pensado para adaptarse a LaTeX. La redaccion evita presentar como implementados metodos que no se usaron: no se incluye SHAP/LIME como metodologia, no se incluye ResUNet y no se presenta 3D U-Net como experimento propio.

## Version estructurada para la memoria

```latex
\section{Metodologia}

El objetivo de la metodologia es describir el protocolo experimental seguido para evaluar modelos de aprendizaje profundo en imagen toracica relacionada con COVID-19. El estudio se organiza en torno a dos modalidades de imagen: radiografia de torax (CXR) y tomografia computarizada (CT). En ambas modalidades se entrenan modelos de clasificacion, se evaluan estrategias de manejo del desbalanceo y se generan explicaciones visuales mediante Grad-CAM. Ademas, se entrenan modelos de segmentacion utilizando las mascaras disponibles y se incorpora un analisis complementario de calibracion probabilistica sobre los clasificadores ya entrenados.

La comparacion entre CXR y CT se plantea como una comparacion metodologica y no como una equivalencia clinica directa. En CXR, las etiquetas corresponden a categorias diagnosticas o radiologicas, mientras que en CT las etiquetas representan grados de severidad radiologica. Asimismo, las mascaras disponibles tienen distinto significado: en CXR delimitan el campo pulmonar, mientras que en CT delimitan regiones de infeccion o lesion. Esta diferencia condiciona tanto el entrenamiento de segmentacion como la interpretacion de la explicabilidad.

\subsection{Diseno general del estudio}

El pipeline experimental se divide en cinco bloques principales: preparacion de datos, clasificacion, segmentacion, explicabilidad y calibracion. En primer lugar, se preparan los datasets CXR y CT, se generan las particiones de entrenamiento, validacion y prueba, y se aplican transformaciones diferenciadas por modalidad. En segundo lugar, se entrenan clasificadores basados en arquitecturas CNN preentrenadas. En tercer lugar, se entrenan modelos de segmentacion U-Net y Attention U-Net sobre las mascaras disponibles. En cuarto lugar, se generan mapas Grad-CAM para analizar la saliencia visual de los clasificadores y compararla con las mascaras. Finalmente, se analiza si la confianza probabilistica de los clasificadores se corresponde con su rendimiento observado.

Todos los resultados documentados en la memoria corresponden a ejecuciones completas. El conjunto de validacion se utiliza para seleccionar configuraciones, aplicar early stopping y ajustar umbrales cuando corresponde. El conjunto de test se reserva para la evaluacion final de cada experimento.

\subsection{Conjuntos de datos}

\subsubsection{COVID-19 Radiography Database}

Para la modalidad CXR se utiliza el dataset COVID-19 Radiography Database. Este conjunto contiene radiografias de torax organizadas en cuatro clases: COVID-19, Lung Opacity, Normal y Viral Pneumonia. En el pipeline de clasificacion, cada imagen se asocia a una de estas cuatro etiquetas, formulando el problema como una clasificacion multiclase.

El dataset tambien incluye mascaras pulmonares. Estas mascaras se utilizan para la tarea de segmentacion pulmonar y como referencia anatomica para evaluar si los mapas Grad-CAM del clasificador se concentran dentro del campo pulmonar. Es importante destacar que estas mascaras no representan lesiones COVID-19 ni opacidades patologicas; delimitan una region anatomica. Por tanto, en CXR la comparacion entre saliencia y mascara mide plausibilidad anatomica, no localizacion de lesion.

El split CXR se realiza de forma estratificada con semilla fija, manteniendo la proporcion de clases en entrenamiento, validacion y prueba. La particion final contiene 14.815 imagenes de entrenamiento, 3.175 de validacion y 3.175 de prueba.

\subsubsection{MosMedData}

Para la modalidad CT se utiliza MosMedData, un dataset de estudios de tomografia computarizada toracica con etiquetas de severidad radiologica. Las clases originales CT-0, CT-1, CT-2, CT-3 y CT-4 se procesan en el pipeline de clasificacion agrupando CT-3 y CT-4 como CT-3+, con el objetivo de reducir la fragmentacion de las clases mas severas y mejorar la estabilidad experimental.

Los estudios CT son volumenes tridimensionales. Para la clasificacion, se extraen slices 2D a partir de cada volumen. Antes de guardar los slices se aplica una ventana de Hounsfield adecuada para tejido pulmonar, con rango $[-1000, 400]$, se normaliza la intensidad a formato imagen y se redimensiona cada slice a $256 \times 256$ pixeles. Se extrae la region central del volumen, correspondiente al intervalo entre el 20\% y el 80\% de los cortes, para reducir slices extremos poco informativos.

La particion CT se realiza por identificador de estudio, no por slice individual. Esta decision es fundamental para evitar fuga de informacion, ya que slices del mismo estudio no deben aparecer simultaneamente en entrenamiento y prueba. Tras la extraccion de slices, la clasificacion CT utiliza 19.456 slices de entrenamiento, 4.170 de validacion y 4.155 de prueba, procedentes de 777, 166 y 167 estudios, respectivamente.

MosMedData tambien proporciona un subconjunto con mascaras de infeccion o lesion. Para la segmentacion CT se construye un dataset especifico a partir de los estudios anotados, extrayendo slices con mascara positiva. Este dataset de segmentacion contiene 508 slices de entrenamiento, 86 de validacion y 110 de prueba, separados por estudio en 35, 7 y 8 estudios, respectivamente.

\subsection{Preprocesamiento y particiones}

El preprocesamiento se adapta a cada modalidad. En CXR, las imagenes se cargan en formato RGB para ser compatibles con arquitecturas preentrenadas sobre ImageNet. En entrenamiento se aplican transformaciones moderadas: redimensionado a $224 \times 224$, volteo horizontal aleatorio, rotacion aleatoria de hasta 15 grados, transformaciones afines suaves, variaciones limitadas de brillo y contraste, conversion a tensor y normalizacion con media y desviacion estandar de ImageNet. En validacion y prueba se aplica unicamente redimensionado, conversion a tensor y normalizacion, sin transformaciones aleatorias.

En CT, los slices se cargan en escala de grises. Para clasificacion se redimensionan a $256 \times 256$, se aplican transformaciones conservadoras durante entrenamiento, incluyendo volteo horizontal, rotacion maxima de 5 grados y transformaciones afines suaves, y se normalizan con media y desviacion estandar de 0,5. En validacion y prueba no se aplican transformaciones aleatorias.

En segmentacion, las transformaciones geometricas se aplican de forma sincronizada a imagen y mascara para mantener la correspondencia espacial. Durante entrenamiento se utilizan volteos horizontales, rotaciones suaves y, en determinadas variantes CT, recortes de entrenamiento basados en parches. En validacion y prueba las imagenes y mascaras se redimensionan de forma determinista al tamano definido para cada modalidad.

\subsection{Clasificacion}

\subsubsection{Arquitecturas evaluadas}

La clasificacion se formula como un problema multiclase. Se comparan tres arquitecturas CNN representativas de distintas familias de transfer learning: ResNet-50, DenseNet-121 y EfficientNet-B0. En todos los casos se emplean pesos preentrenados y se sustituye la cabeza clasificadora final para adaptarla al numero de clases de cada modalidad.

En CXR, las imagenes se procesan como entradas RGB de tres canales. En CT, los slices se procesan como imagenes de un canal. Para adaptar modelos preentrenados originalmente con tres canales, se modifica la primera convolucion cuando es necesario, promediando los pesos preentrenados de los tres canales para inicializar una convolucion de un canal. De este modo se aprovecha parcialmente la informacion aprendida durante el preentrenamiento sin replicar artificialmente el slice CT en tres canales.

\subsubsection{Transfer learning y fine-tuning}

El entrenamiento de clasificacion se realiza en dos fases. En la primera fase se congela el backbone y se entrena solo la cabeza clasificadora durante 5 epocas. En la segunda fase se descongela el modelo completo y se realiza fine-tuning durante las epocas restantes con una tasa de aprendizaje menor. La tasa de aprendizaje inicial es $10^{-4}$ y durante fine-tuning se utiliza una tasa diez veces menor. Se emplea el optimizador AdamW con weight decay $10^{-5}$.

El entrenamiento utiliza early stopping basado en validacion, con paciencia de 5 epocas, y un scheduler ReduceLROnPlateau que reduce la tasa de aprendizaje cuando la perdida de validacion deja de mejorar. El tamano de batch en clasificacion es 32 y la semilla aleatoria se fija a 42 para mejorar la reproducibilidad.

\subsubsection{Manejo del desbalanceo en clasificacion}

Para estudiar el efecto del desbalanceo de clases, cada arquitectura se entrena bajo varias estrategias:

\begin{itemize}
  \item \textbf{Baseline}: entrenamiento con CrossEntropyLoss sin correccion explicita.
  \item \textbf{Weighted cross-entropy}: entropia cruzada con pesos inversamente proporcionales a la frecuencia de cada clase.
  \item \textbf{Focal loss}: perdida focal con $\gamma=2$, usando pesos de clase como factor $\alpha$.
  \item \textbf{Oversampling}: muestreo ponderado mediante WeightedRandomSampler, aumentando la probabilidad de seleccionar muestras de clases minoritarias durante el entrenamiento.
\end{itemize}

La augmentacion se considera una tecnica de regularizacion y aumento de variabilidad visual durante entrenamiento, no una fuente de nuevos casos clinicos reales.

\subsection{Segmentacion}

\subsubsection{Tareas de segmentacion}

La segmentacion se estudia en CXR y CT con significados distintos. En CXR, el objetivo es segmentar el campo pulmonar utilizando mascaras pulmonares. En CT, el objetivo es segmentar regiones de infeccion o lesion utilizando las mascaras patologicas disponibles en MosMedData. Esta diferencia es esencial: la segmentacion pulmonar no equivale a segmentacion de COVID-19, mientras que la segmentacion CT se aproxima de forma mas directa a la localizacion patologica.

\subsubsection{Arquitecturas de segmentacion}

Se implementan dos arquitecturas de segmentacion binaria: U-Net y Attention U-Net. Ambas siguen una estructura encoder-decoder con conexiones de salto. U-Net concatena caracteristicas del encoder con las del decoder para recuperar informacion espacial fina. Attention U-Net incorpora compuertas de atencion en las conexiones de salto, filtrando las caracteristicas transferidas desde el encoder antes de combinarlas con el decoder.

La salida de los modelos es un mapa de logits de un canal. Durante la inferencia se aplica una sigmoid para obtener probabilidades por pixel y posteriormente se binariza la mascara mediante un umbral. Las configuraciones base utilizan $base\_features=32$, lo que genera una progresion de filtros $(32, 64, 128, 256)$ en el encoder.

\subsubsection{Entrenamiento y perdidas}

El entrenamiento de segmentacion utiliza AdamW con tasa de aprendizaje $10^{-4}$, weight decay $10^{-5}$, batch size base 8 y hasta 30 epocas en los modelos base. Se guarda el mejor modelo segun Dice de validacion y se aplica early stopping. Las metricas principales son Dice e IoU, mientras que pixel accuracy se reporta como metrica secundaria. Esta ultima debe interpretarse con cautela, especialmente en CT, porque el fondo domina la mayoria de los pixeles.

Como perdida base se utiliza una combinacion de binary cross-entropy con logits y Dice loss. Para CT se exploran variantes con weighted Tversky + BCE, incorporando `pos_weight` para aumentar la penalizacion de errores sobre pixeles positivos. En las variantes Tversky se emplean $\alpha=0,3$ y $\beta=0,7$, dando mayor penalizacion a falsos negativos, una decision razonable en lesiones pequenas donde perder regiones positivas puede degradar fuertemente Dice e IoU.

\subsubsection{Variantes CT y estudio de ablacion}

La segmentacion CT presenta mayor dificultad que la segmentacion pulmonar CXR debido al reducido tamano de las lesiones, el fuerte desbalance pixel a pixel y la variabilidad entre estudios. Por ello se disena un estudio de ablacion progresivo sobre Attention U-Net:

\begin{itemize}
  \item \textbf{Baseline}: U-Net y Attention U-Net con perdida Dice+BCE.
  \item \textbf{Tversky ponderada}: uso de weighted Tversky+BCE con `pos_weight`.
  \item \textbf{Entrenamiento por parches}: entrenamiento con recortes de la imagen para aumentar la frecuencia de regiones informativas.
  \item \textbf{Positive crop sampling}: seleccion preferente de parches que contienen lesion.
  \item \textbf{Mixed context training}: combinacion de contexto local mediante parches y evaluacion sobre slices completos.
  \item \textbf{Seleccion de umbral en validacion}: busqueda del umbral que maximiza Dice en validacion antes de evaluar en test.
  \item \textbf{2.5D CT}: uso del slice anterior, slice actual y slice posterior como tres canales de entrada.
  \item \textbf{Ensemble}: promedio ponderado de probabilidades entre dos modelos CT complementarios, seleccionando peso y umbral en validacion.
  \item \textbf{Ajuste de capacidad}: aumento de `base_features` a 32 en la variante mas prometedora.
\end{itemize}

El enfoque 2.5D se considera una exploracion metodologica. Su objetivo es introducir contexto volumetrico local sin entrenar una red 3D completa. No se presenta como arquitectura principal, sino como una variante dentro del estudio experimental CT.

\subsection{Explicabilidad mediante Grad-CAM}

La explicabilidad se aborda mediante Grad-CAM, aplicado sobre clasificadores CNN previamente entrenados. Grad-CAM calcula un mapa de saliencia utilizando los gradientes de una clase objetivo respecto a activaciones de una capa convolucional profunda. Las capas objetivo se seleccionan segun la arquitectura: el ultimo bloque convolucional en ResNet-50, el ultimo bloque denso en DenseNet-121 y el ultimo bloque de features en EfficientNet-B0.

Los mapas Grad-CAM se normalizan al rango $[0,1]$ y se redimensionan al tamano de entrada. Para compararlos con mascaras se binariza la saliencia seleccionando las regiones de mayor activacion, usando el cuantil 0,80 como umbral. Se calculan tres indicadores:

\begin{itemize}
  \item IoU entre la saliencia binarizada y la mascara de referencia.
  \item Proporcion de saliencia total que cae dentro de la mascara.
  \item Indicador de si el punto de maxima activacion se encuentra dentro de la mascara.
\end{itemize}

La interpretacion depende de la modalidad. En CXR, la mascara de referencia es pulmonar, por lo que la comparacion evalua si el clasificador atiende a una region anatomica plausible. En CT, la mascara de referencia corresponde a infeccion o lesion, por lo que la comparacion evalua alineacion con una region patologica anotada. En ningun caso Grad-CAM se interpreta como una segmentacion ni como una prueba causal.

\subsection{Calibracion probabilistica}

De forma complementaria, se analiza la calibracion probabilistica de los clasificadores ya entrenados. Este analisis no reentrena modelos, sino que utiliza los CSV de predicciones guardados durante la evaluacion. El objetivo es estudiar si la confianza asignada por el modelo se corresponde con su probabilidad real de acierto.

Las metricas calculadas son:

\begin{itemize}
  \item confianza maxima media;
  \item Expected Calibration Error (ECE);
  \item Maximum Calibration Error (MCE);
  \item Brier score multiclase;
  \item negative log-likelihood;
  \item numero y proporcion de errores de alta confianza, definidos con confianza mayor o igual a 0,90.
\end{itemize}

Ademas, se generan diagramas de fiabilidad e histogramas de confianza para visualizar la relacion entre confianza predicha y accuracy observada. Esta fase permite complementar las metricas clasicas, ya que en imagen medica no solo importa si el modelo acierta, sino tambien si sus predicciones incorrectas aparecen con alta seguridad.

\subsection{Metricas de evaluacion}

\subsubsection{Metricas de clasificacion}

En clasificacion se reportan accuracy, precision, recall, F1-score por clase, F1-macro, F1-weighted, matriz de confusion y AUC macro one-vs-rest. F1-macro se considera especialmente relevante porque da el mismo peso a todas las clases y permite evaluar mejor el comportamiento en clases minoritarias. Tambien se calculan intervalos de confianza mediante bootstrap para accuracy y F1-macro, y se aplica una comparacion de McNemar entre los dos mejores clasificadores de cada modalidad cuando procede.

\subsubsection{Metricas de segmentacion}

En segmentacion se priorizan Dice e IoU/Jaccard, ya que ambas cuantifican el solapamiento entre mascara predicha y mascara real. Pixel accuracy se reporta como metrica secundaria, pero no se utiliza como criterio principal en CT debido al fuerte desbalance entre fondo y lesion.

\subsubsection{Metricas de explicabilidad}

La explicabilidad se evalua cuantitativamente mediante IoU saliencia-mascara, proporcion de saliencia dentro de la mascara y tasa de maxima activacion dentro de la mascara. Estas metricas no validan clinicamente el modelo, pero aportan una medida reproducible de plausibilidad espacial.

\subsubsection{Metricas de calibracion}

La calibracion se evalua con ECE, MCE, Brier score, negative log-likelihood y errores de alta confianza. Estas metricas permiten detectar modelos que, aun teniendo buen rendimiento, pueden asignar confianza excesiva a predicciones incorrectas.

\subsection{Diseno experimental y reproducibilidad}

La reproducibilidad se aborda mediante una configuracion centralizada de rutas, tamanos de imagen, hiperparametros base y semilla aleatoria. Los experimentos guardan artefactos en disco, incluyendo checkpoints, historiales de entrenamiento, resumenes JSON, predicciones CSV, informes de clasificacion, matrices de confusion, metricas de segmentacion, mapas Grad-CAM, figuras cualitativas y metricas de calibracion.

Los resultados finales se consolidan a partir de estos artefactos mediante scripts y notebooks especificos. La seleccion de modelos, umbrales y pesos de ensemble se realiza sobre validacion. La evaluacion final se realiza sobre test reservado. Este diseno evita ajustar decisiones metodologicas directamente sobre el conjunto de prueba y permite trazar cada resultado hasta el experimento que lo genero.
```

## Datos metodologicos verificados

| Bloque | Decision real |
|---|---|
| Semilla | `42` |
| CXR image size | `224x224` |
| CT image size | `256x256` |
| CXR split | `14815 train`, `3175 val`, `3175 test` |
| CT clasificacion split | `19456 train`, `4170 val`, `4155 test` slices |
| CT clasificacion estudios | `777 train`, `166 val`, `167 test` estudios |
| CT segmentacion split | `508 train`, `86 val`, `110 test` slices |
| CT segmentacion estudios | `35 train`, `7 val`, `8 test` estudios |
| Clasificacion batch | `32` |
| Clasificacion epochs | `20` total: `5` head + `15` fine-tuning |
| Clasificacion optimizer | `AdamW`, learning rate `1e-4`, weight decay `1e-5` |
| Segmentacion base | `30` epochs, batch `8`, learning rate `1e-4`, weight decay `1e-5` |
| XAI | Grad-CAM, saliency quantile `0.80` |
| Calibracion | ECE, MCE, Brier, NLL, errores con confianza `>=0.90` |

## Advertencias para no cometer incoherencias

- No escribir que CXR segmenta lesion COVID: segmenta pulmon.
- No escribir que CT y CXR tienen etiquetas equivalentes.
- No escribir que SHAP/LIME fueron implementados.
- No escribir que ResUNet fue usado.
- No escribir que 3D U-Net fue entrenado.
- Presentar 2.5D como exploracion CT, no como enfoque principal.
- Presentar `bf32` como `base_features=32`, no como bfloat32.
- Reportar solo ejecuciones completas en la memoria.
