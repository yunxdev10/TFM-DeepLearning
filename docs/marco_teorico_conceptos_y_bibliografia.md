# Marco teorico: conceptos tecnicos y referencias para el TFM

Fecha: 2026-05-17

## Objetivo del documento

Este documento reune los conceptos tecnicos usados en el proyecto para servir como base del Estado del Arte, Metodologia, Resultados y Discusion. La idea es que puedas nombrar y redactar desde conceptos basicos hasta decisiones concretas del pipeline:

- imagen medica CXR y CT;
- clasificacion multi-clase;
- segmentacion semantica;
- transfer learning;
- manejo del desbalanceo;
- muestreo ponderado y seleccion de batches;
- seleccion de umbrales en validacion;
- entrenamiento por patches, contexto mixto y 2.5D;
- ensembles y estudios de ablacion;
- metricas de evaluacion;
- explicabilidad con Grad-CAM;
- limitaciones metodologicas;
- referencias bibliograficas base.

Documento complementario para estudiar y redactar con mas profundidad: `docs/guia_lectura_conceptos_tfm.md`.

## 1. Contexto clinico y de imagen medica

### COVID-19 en imagen medica

COVID-19 es una enfermedad infecciosa respiratoria que puede producir afectacion pulmonar visible en imagen medica. En aprendizaje profundo se han usado principalmente dos modalidades:

- **Radiografia de torax (CXR)**: imagen 2D rapida, barata y frecuente. Tiene menor detalle anatomico que CT, pero permite estudios grandes.
- **Tomografia computarizada (CT)**: volumen 3D compuesto por cortes/slices. Tiene mayor detalle anatomico y permite observar patrones como opacidades en vidrio deslustrado o consolidaciones, pero suele tener menos datos anotados y mas coste.

En el TFM se trabaja con dos tareas distintas:

- CXR: clasificacion de imagenes en 4 clases: COVID, Lung Opacity, Normal, Viral Pneumonia.
- CT: clasificacion de severidad en grupos CT-0, CT-1, CT-2, CT-3+.

Esta diferencia es importante: las etiquetas CXR son categorias diagnosticas, mientras que las etiquetas CT representan severidad. Por tanto, la comparacion CXR vs CT debe interpretarse como comparacion metodologica, no como equivalencia clinica directa.

### CXR

Una CXR es una proyeccion 2D del torax. Contiene pulmones, mediastino, costillas, diafragma y tejidos blandos. En deep learning, una CXR puede sufrir sesgos por:

- marcas laterales o texto;
- diferencias de adquisicion entre hospitales;
- presencia de dispositivos;
- variacion de posicionamiento;
- correlaciones no patologicas.

Por eso la explicabilidad es util: permite comprobar si el modelo mira regiones razonables.

### CT

Una CT se compone de multiples slices. En este proyecto se usa una aproximacion 2D: cada slice se trata como imagen independiente para clasificacion/segmentacion. Esta eleccion es realista para recursos locales, pero introduce limitaciones:

- se pierde contexto volumetrico 3D;
- slices de un mismo estudio son muy parecidos;
- hay riesgo de fuga de datos si se separa por slice y no por paciente/estudio;
- algunas lesiones son pequenas y dificiles de segmentar.

Por eso el proyecto usa split por `study_id` en CT.

## 2. Datasets y anotaciones

### Dataset CXR

El dataset CXR usado procede de la COVID-19 Radiography Database publicada en Kaggle por el grupo de Qatar University y University of Dhaka. Contiene imagenes de radiografia organizadas en clases como COVID-19, Lung Opacity, Normal y Viral Pneumonia. En este proyecto tambien se aprovechan las mascaras pulmonares asociadas. Las mascaras se usan para:

- entrenar segmentacion pulmonar;
- evaluar si Grad-CAM cae dentro del campo pulmonar.

Importante: la mascara CXR usada aqui es pulmonar, no una mascara de lesion COVID. Por tanto, una alta alineacion saliencia-pulmon indica plausibilidad anatomica, pero no demuestra que el modelo localice una lesion. Para redactar el estado del arte conviene separar tres ideas: el repositorio de datos CXR, los trabajos de clasificacion COVID con CXR y los trabajos especificos de localizacion/segmentacion en CXR.

### Dataset CT

El dataset CT usado procede de MosMedData. Incluye estudios CT por severidad y un subconjunto con mascaras de infeccion. Las mascaras CT se usan para:

- segmentacion de lesion/infeccion;
- evaluacion Grad-CAM frente a region patologica.

La anotacion CT es mucho mas escasa que CXR. Esto explica por que la segmentacion CT y la explicabilidad CT son mas dificiles.

### Etiqueta

Una etiqueta o `label` es la clase asociada a una imagen. Puede representar:

- diagnostico: COVID, Normal, Viral Pneumonia;
- patron radiologico: Lung Opacity;
- severidad: CT-0, CT-1, CT-2, CT-3+.

### Mascara

Una mascara es una imagen binaria donde los pixeles positivos indican una region de interes:

- en CXR: region pulmonar;
- en CT: region de infeccion/lesion.

En segmentacion, la mascara es el objetivo a predecir. En XAI, la mascara sirve como referencia para medir si la saliencia cae en una zona relevante.

## 3. Preprocesamiento y preparacion de datos

### Resize

Consiste en ajustar todas las imagenes a un tamano fijo para que entren en la red neuronal:

- CXR: 224 x 224, compatible con modelos ImageNet;
- CT/segmentacion: 256 x 256.

### Normalizacion

La normalizacion transforma intensidades para estabilizar el entrenamiento. En modelos preentrenados en ImageNet se usan medias y desviaciones tipicas de ImageNet. En CT de un canal se uso normalizacion simple de escala gris.

### Windowing HU en CT

En CT, las intensidades representan unidades Hounsfield. El windowing recorta y reescala un rango de interes. Para pulmon suele usarse una ventana aproximada `[-1000, 400]`, que resalta parenquima pulmonar y lesiones.

### Data augmentation

Augmentation genera variaciones artificiales para mejorar generalizacion:

- flip horizontal;
- rotacion;
- affine/traslacion;
- cambios suaves de contraste/brillo en CXR.

En imagen medica se debe ser conservador: una transformacion demasiado fuerte puede crear anatomia no realista.

### Split train/val/test

El dataset se separa en:

- **train**: ajuste de pesos;
- **validation**: seleccion de hiperparametros, early stopping y thresholds;
- **test**: evaluacion final.

En CT es clave hacer split por `study_id`, no por slice, para evitar que slices del mismo paciente aparezcan en train y test.

### Split estratificado

Un split estratificado intenta mantener proporciones similares de clases en train, validation y test. En CXR se usa porque las clases tienen tamanos diferentes. En CT se combina con el split por estudio: primero se separan estudios, manteniendo distribucion de severidad, y despues se expanden los estudios a sus slices.

### Fusion de clases CT-3 y CT-4

MosMedData contiene clases de severidad CT-0, CT-1, CT-2, CT-3 y CT-4. En el TFM se agrupan CT-3 y CT-4 como `CT-3+`. Esta decision reduce fragmentacion en las clases mas graves, aumenta el numero de ejemplos por clase final y hace mas estable la evaluacion multiclase. Debe explicarse como una decision metodologica de agregacion de severidad, no como una nueva etiqueta clinica original.

### Extraccion de slices CT

Los estudios CT originales son volumenes NIfTI. Para trabajar con modelos 2D se extraen cortes axiales y se guardan como PNG. En este proyecto se usan slices centrales aproximados, evitando extremos del volumen donde suele haber menos tejido pulmonar. Esto reduce ruido, pero tambien implica que el enfoque no explota toda la informacion 3D.

### Ejecuciones full

En la version final del proyecto, los notebooks y scripts experimentales trabajan en modo `full`. Esto significa que las tablas, graficas y conclusiones de la memoria se basan en entrenamientos completos, con las particiones train/validation/test definidas para cada modalidad.

Esta decision evita mezclar pruebas tecnicas reducidas con evidencia cientifica. Para que un resultado sea reportable en el TFM debe proceder de una ejecucion completa, guardar sus artefactos de forma reproducible y evaluarse sobre el conjunto de test correspondiente.

### Reproducibilidad

La reproducibilidad consiste en poder repetir el experimento con la misma configuracion y obtener resultados comparables. En este proyecto se usa:

- semilla fija (`RANDOM_SEED = 42`);
- configuracion centralizada;
- guardado de hiperparametros;
- guardado de historiales de entrenamiento;
- guardado de predicciones y summaries JSON/CSV.

En deep learning, la reproducibilidad no siempre es exacta entre CPU, GPU o MPS, pero el registro de configuracion permite defender el experimento.

## 4. Deep learning y CNN

### Red neuronal

Una red neuronal aprende una funcion que transforma entradas en salidas. En este TFM:

- entrada: imagen CXR o CT;
- salida clasificacion: probabilidades de clase;
- salida segmentacion: mascara por pixel.

### CNN

Una Convolutional Neural Network usa capas convolucionales para extraer patrones espaciales:

- bordes;
- texturas;
- formas;
- regiones anatomicas.

Las CNN son adecuadas para imagen porque comparten pesos y preservan estructura local.

### Feature extraction

Las primeras capas suelen detectar patrones simples; las capas profundas detectan conceptos mas abstractos. En transfer learning se reutilizan estas representaciones aprendidas.

## 5. Transfer learning y fine-tuning

### Transfer learning

Transfer learning reutiliza un modelo preentrenado en un dataset grande, normalmente ImageNet, para una tarea medica con menos datos. Ventajas:

- acelera convergencia;
- mejora rendimiento con pocos datos;
- reduce necesidad de entrenar desde cero.

Limitacion: ImageNet contiene imagenes naturales, no imagen medica. Por eso el fine-tuning sigue siendo necesario.

### Fine-tuning

Fine-tuning consiste en adaptar pesos preentrenados al nuevo dominio. En este proyecto:

1. se congela backbone y se entrena la cabeza;
2. se descongelan capas para ajustar todo el modelo con learning rate menor.

### Backbone y cabeza clasificadora

- **Backbone**: parte convolucional que extrae caracteristicas.
- **Head/classifier**: capa final que convierte caracteristicas en clases.

## 6. Arquitecturas de clasificacion

### ResNet-50

ResNet introduce conexiones residuales. En vez de aprender directamente una funcion completa, aprende un residual. Esto facilita entrenar redes profundas y reduce el problema de degradacion al aumentar capas.

Conceptos clave:

- residual block;
- skip connection;
- gradientes mas estables;
- arquitectura robusta como baseline.

En CT fue el mejor por accuracy.

### DenseNet-121

DenseNet conecta cada capa con todas las capas posteriores dentro de un bloque denso. Esto favorece reutilizacion de caracteristicas y flujo de gradiente.

Conceptos clave:

- dense connectivity;
- feature reuse;
- menos parametros que redes muy anchas;
- buen rendimiento en tareas medicas.

En CXR fue el mejor modelo global con weighted cross-entropy. En CT tuvo mejor F1-macro/AUC que ResNet.

### EfficientNet-B0

EfficientNet escala profundidad, anchura y resolucion de forma compuesta. Busca buena relacion rendimiento-coste.

Conceptos clave:

- compound scaling;
- eficiencia computacional;
- arquitectura compacta.

En este proyecto fue competitivo, pero no el mejor final.

## 7. Desbalanceo de clases

### Problema

Un dataset esta desbalanceado cuando algunas clases tienen muchos mas ejemplos que otras. Esto puede hacer que el modelo prediga clases mayoritarias y falle en minoritarias.

En CT, las clases graves tienen menos muestras, lo que afecta sensibilidad y F1-macro.

### Baseline sin balanceo

El baseline usa `CrossEntropyLoss` sin pesos especiales y muestreo aleatorio estandar. Es imprescindible porque permite comprobar si una estrategia de balanceo realmente mejora algo. En el TFM, el baseline CT fue muy fuerte: algunas estrategias de balanceo no lo superaron, lo cual es un resultado metodologico importante.

### Pesos de clase por frecuencia inversa

Los pesos de clase se calculan a partir de la frecuencia de cada clase:

`peso_clase = total_muestras / (numero_clases * muestras_de_la_clase)`

Asi, una clase minoritaria recibe mas peso durante el calculo de la perdida. Esto no crea datos nuevos; cambia la penalizacion de los errores.

### Weighted cross-entropy

Asigna mayor peso a clases menos frecuentes. En CXR fue la mejor estrategia global.

En la memoria conviene explicar que weighted CE actua sobre la funcion de perdida: si el modelo falla una clase minoritaria, la penalizacion es mayor que con cross-entropy normal.

### Focal loss

Focal loss reduce el peso de ejemplos faciles y se centra en ejemplos dificiles. Fue propuesta para deteccion de objetos, pero se usa tambien en clasificacion desbalanceada.

En este proyecto no mejoro claramente CT.

Conceptos internos:

- `alpha`: peso de clase, usado para compensar desbalance;
- `gamma`: controla cuanto se reduce la contribucion de ejemplos faciles;
- `pt`: probabilidad asignada a la clase correcta.

### Oversampling

Consiste en muestrear mas veces ejemplos de clases minoritarias. Puede ayudar, pero tambien puede aumentar sobreajuste si los datos minoritarios son pocos.

En el codigo se implementa con `WeightedRandomSampler`: cada muestra recibe un peso inversamente proporcional a la frecuencia de su clase y el `DataLoader` puede repetir muestras minoritarias con reemplazo. A diferencia de weighted CE, oversampling cambia la distribucion de los batches, no la formula de la perdida.

### Diferencia entre weighted CE y oversampling

Ambas tecnicas buscan compensar desbalance, pero actuan en lugares distintos:

- weighted CE: modifica la perdida;
- oversampling: modifica la probabilidad de ver ejemplos minoritarios durante entrenamiento;
- focal loss: modifica la perdida en funcion de dificultad del ejemplo.

Por eso se pueden comparar como estrategias experimentales separadas.

### Desbalance pixel a pixel en segmentacion

En segmentacion CT el desbalance no es solo por numero de imagenes: tambien ocurre dentro de cada imagen. La lesion ocupa muy pocos pixeles frente al fondo. Por eso pixel accuracy puede ser alta aunque la prediccion sea pobre, y por eso se usan Dice, IoU, Tversky, `pos_weight` y muestreo de parches centrado en positivos.

### `pos_weight` en BCE de segmentacion

`pos_weight` aumenta la penalizacion de los pixeles positivos mal clasificados. En el TFM se estima como proporcion aproximada entre pixeles negativos y positivos, con un maximo para evitar pesos extremos. Esto ayuda cuando la mascara de lesion es muy pequena.

## 8. Entrenamiento y optimizacion

### Loss function

La funcion de perdida mide el error que el modelo debe minimizar.

- Clasificacion: CrossEntropy, Focal Loss.
- Segmentacion: BCE, Dice/Tversky combinadas.

### Logits, softmax y sigmoid

El modelo no produce directamente una etiqueta final. Produce **logits**, valores sin normalizar:

- en clasificacion multiclase, los logits se transforman con `softmax` para obtener probabilidades por clase;
- en segmentacion binaria, cada pixel produce un logit que se transforma con `sigmoid` para obtener probabilidad de pertenecer a la mascara.

Esta diferencia explica por que clasificacion usa `CrossEntropyLoss` y segmentacion usa BCE con logits combinada con Dice/Tversky.

### AdamW

AdamW es una variante de Adam con weight decay desacoplado. Suele ser estable en deep learning moderno.

### Weight decay

Weight decay penaliza pesos demasiado grandes. Funciona como regularizacion y ayuda a reducir sobreajuste.

### Learning rate

Controla cuanto se actualizan los pesos. Un valor alto puede inestabilizar; uno bajo puede entrenar muy lento.

### Batch size

El batch size define cuantas imagenes se procesan antes de actualizar los pesos. Un batch mayor suele estabilizar gradientes pero consume mas memoria. En CT se usaron batch sizes mas pequenos en variantes de mayor capacidad o con patches grandes para ajustarse a la memoria disponible.

### Early stopping

Detiene el entrenamiento si la validacion deja de mejorar. Reduce sobreajuste y ahorra tiempo.

### ReduceLROnPlateau

Reduce el learning rate cuando la metrica de validacion se estanca.

### Entrenamiento en dos fases

En clasificacion con modelos preentrenados se usa una estrategia de dos fases:

1. congelar el backbone y entrenar solo la cabeza clasificadora;
2. descongelar el modelo completo y hacer fine-tuning con learning rate menor.

Esto reduce el riesgo de destruir de golpe las representaciones preentrenadas de ImageNet.

### Hiperparametros

Un hiperparametro es una decision externa al aprendizaje de pesos: learning rate, batch size, numero de epocas, pesos de la loss, threshold, tamano de patch, probabilidad de crop positivo o numero de filtros base. En este TFM la experimentacion consiste precisamente en modificar hiperparametros y comparar su efecto con validacion/test.

### Hardware: CPU, GPU, MPS y tiempo de entrenamiento

El hardware afecta sobre todo al tiempo de entrenamiento. CPU, M1/MPS, M4/MPS o GPU CUDA pueden producir pequenas diferencias numericas por implementaciones internas, pero no deberian cambiar de forma sustancial la conclusion si el pipeline, datos, seeds y configuracion son los mismos. Para la memoria, lo importante es reportar entorno y reconocer que los resultados se obtienen bajo una configuracion reproducible.

## 9. Metricas de clasificacion

### Accuracy

Proporcion de predicciones correctas. Es intuitiva, pero puede ser enganosa con clases desbalanceadas.

### Precision

De los casos que el modelo predice como una clase, cuantos son correctos.

### Recall o sensibilidad

De los casos reales de una clase, cuantos detecta el modelo.

En medicina, recall suele ser importante porque interesa no perder casos positivos.

### F1-score

Media armonica de precision y recall. Es util cuando hay desbalance.

### F1-macro

Promedia F1 por clase dando el mismo peso a cada clase. Es importante en CT porque penaliza mal rendimiento en clases minoritarias.

### F1-weighted

Promedia F1 ponderando por soporte de clase. Puede ser alto aunque clases minoritarias vayan mal.

### Matriz de confusion

Tabla que muestra clases reales vs clases predichas. Permite ver que clases se confunden entre si.

### ROC-AUC multiclase

Evalua separacion probabilistica. En multiclase suele calcularse one-vs-rest y promediar.

### Intervalo de confianza bootstrap

El bootstrap remuestrea predicciones para estimar incertidumbre de metricas. En Fase 4 se usa para accuracy y F1-macro.

### McNemar

Prueba estadistica para comparar dos clasificadores sobre los mismos ejemplos. Usa discordancias:

- n01: modelo A falla y B acierta;
- n10: modelo A acierta y B falla.

En los resultados finales, las diferencias top-2 no fueron estadisticamente concluyentes.

## 10. Segmentacion semantica

### Segmentacion

La segmentacion asigna una clase a cada pixel. En este proyecto:

- CXR: pulmon vs fondo;
- CT: lesion/infeccion vs fondo.

### U-Net

U-Net es una arquitectura encoder-decoder con skip connections. El encoder captura contexto; el decoder recupera resolucion espacial. Las conexiones skip ayudan a reconstruir detalles.

### Attention U-Net

Attention U-Net anade mecanismos de atencion en las conexiones skip para enfatizar regiones relevantes y reducir ruido.

En este proyecto fue la mejor familia en CXR y CT.

### Encoder-decoder

Un modelo encoder-decoder comprime la imagen para capturar contexto y despues reconstruye una salida con resolucion espacial. En segmentacion, esta estructura permite combinar contexto global y detalle local.

### Skip connections en segmentacion

Las skip connections conectan capas tempranas del encoder con capas del decoder. Ayudan a recuperar bordes y detalles que se perderian al comprimir la imagen.

### BCE

Binary Cross-Entropy mide error pixel a pixel en segmentacion binaria.

### Dice loss

Basada en Dice coefficient. Es util cuando la region positiva es pequena.

### Tversky loss

Generaliza Dice permitiendo ponderar falsos positivos y falsos negativos. Es especialmente util en segmentacion medica desbalanceada.

En el TFM se usa Tversky combinada con BCE. Ajustar `alpha` y `beta` permite decidir si se penalizan mas los falsos positivos o los falsos negativos. Esto es relevante porque en CT aparecian dos problemas opuestos: no detectar lesiones pequenas y sobresegmentar regiones de fondo.

### Threshold

La red produce probabilidades. El threshold convierte probabilidades en mascara binaria. Elegir threshold en validacion es importante para no ajustar al test.

En los mejores experimentos CT el threshold subio respecto al valor clasico 0.5. Esto indica que el modelo generaba probabilidades relativamente extensas y necesitaba una decision mas conservadora para reducir falsos positivos.

### Busqueda de threshold en validacion

La busqueda de threshold prueba varios cortes posibles sobre el conjunto de validacion y selecciona el que maximiza Dice. Despues se aplica una unica vez al test. Esto evita elegir el threshold mirando el resultado de test, que seria fuga metodologica.

### Postprocesado

Puede incluir eliminar componentes pequenas o ajustar umbrales. En CT, el postprocesado mejoro poco; la limitacion principal estaba en datos/modelo, no solo en threshold.

### Componentes conectados

Una componente conectada es un grupo de pixeles positivos contiguos. En postprocesado se puede eliminar componentes muy pequenas (`min_component_area`) o quedarse con la mayor. En el TFM se probo esta idea, pero la mejora fue marginal; por tanto se reporta como analisis complementario, no como solucion principal.

### Patch-based training

El entrenamiento por patches recorta regiones de la imagen para que el modelo vea mas detalle local. En CT se probo con patches de 128, 160 y 192. Patches pequenos enfocan lesiones, pero pueden perder contexto anatomico; patches grandes conservan mas contexto, pero reducen el zoom local.

### Positive crop sampling

El muestreo de crops positivos fuerza que una proporcion de parches se centre en pixeles de lesion. Esto combate el problema de que la mayoria de crops aleatorios contengan solo fondo. En el nombre de variantes como `pos70` o `pos80`, el numero indica la probabilidad aproximada de elegir un crop centrado en mascara positiva.

### Mixed context training

El mixed context combina muestras/crops con diferente grado de contexto. La idea es equilibrar:

- contexto global: localizacion anatomica y forma del pulmon;
- detalle local: bordes y pequenas lesiones.

En CT fue una de las mejoras importantes frente al patch puro, porque el patch puro podia perder referencia anatomica.

### 2.5D en CT

El enfoque 2.5D usa varios slices vecinos como canales de entrada, por ejemplo slice anterior, actual y posterior. No es un modelo 3D completo, pero introduce algo de contexto volumetrico. En este TFM se probo como experimento CT; aporto pixel accuracy alta pero no supero al mejor modelo 2D en Dice.

### Slices negativos

Los slices negativos son cortes sin lesion anotada. Incluirlos puede ensenar al modelo a no segmentar fondo como lesion, reduciendo falsos positivos. Pero si se incluyen demasiados, el modelo puede volverse demasiado conservador y perder sensibilidad. Por eso se plantea como experimento de datos, no como garantia de mejora.

### Capacidad del modelo y `base_features`

`base_features` controla el numero inicial de filtros en U-Net/Attention U-Net. Aumentarlo incrementa capacidad del modelo:

- mas capacidad puede capturar patrones complejos;
- tambien aumenta coste y riesgo de sobreajuste.

La variante `bf32` significa `base_features=32`, no bfloat32. Fue el mejor resultado CT, lo que sugiere que la variante ligera anterior estaba limitada por capacidad.

### Ensemble por promedio de probabilidades

Un ensemble combina predicciones de varios modelos. En el TFM se probo un promedio ponderado de probabilidades entre dos modelos CT complementarios y se seleccionaron pesos/threshold en validacion. El ensemble mejoro al mejor modelo individual previo, pero despues fue superado por la variante de mayor capacidad `bf32`.

### Ablation study

Un estudio de ablacion modifica una parte del sistema y mantiene el resto lo mas constante posible para medir su efecto. En el TFM hay ablaciones de patch size, proporcion de crop positivo, contexto 2.5D, capacidad (`base_features`) y ensemble. Esto es muy valioso para defender que no solo se entreno un modelo final, sino que se experimento de forma controlada.

## 11. Metricas de segmentacion

### Dice coefficient

Mide solapamiento entre prediccion y mascara real. Valor 1 es perfecto; 0 indica sin solapamiento.

Es sensible a lesiones pequenas: pocos pixeles mal clasificados pueden hundir el Dice.

### IoU / Jaccard

Intersection over Union compara interseccion frente a union. Suele ser mas estricto que Dice.

### Pixel accuracy

Proporcion de pixeles correctos. En segmentacion desbalanceada puede ser enganosa: si la lesion ocupa pocos pixeles, predecir fondo en casi todo puede dar accuracy alta.

## 12. Explicabilidad XAI

### Explicabilidad

XAI busca explicar que partes de la entrada influyen en la prediccion. En imagen medica es importante para:

- auditar si el modelo usa regiones anatomicas plausibles;
- detectar atajos o sesgos;
- apoyar confianza interpretativa.

### Grad-CAM

Grad-CAM usa gradientes de la clase objetivo respecto a activaciones convolucionales profundas. Produce un mapa de calor indicando regiones importantes para la prediccion.

Ventajas:

- no requiere modificar el modelo;
- funciona con CNN;
- es visualmente interpretable;
- coste moderado.

Limitaciones:

- baja resolucion espacial;
- mapas gruesos;
- no equivale a segmentacion;
- puede ser inestable o poco localizado en lesiones pequenas.

### Saliencia binaria

Para comparar Grad-CAM con una mascara, se binariza el mapa de calor usando un threshold o percentil. En este proyecto se usa cuantile alto para seleccionar zonas mas salientes.

### IoU saliencia-mascara

Mide si el mapa de saliencia cae sobre la mascara disponible:

- CXR: saliencia vs pulmon;
- CT: saliencia vs lesion.

La interpretacion es distinta:

- CXR: plausibilidad anatomica;
- CT: alineacion patologica.

### Hallazgo XAI del TFM

Grad-CAM en CXR muestra alineacion parcial con pulmon. En CT, Grad-CAM no se alinea bien con las mascaras de infeccion. Esto es un hallazgo negativo defendible: un modelo puede acertar la clase sin demostrar que mira la lesion anotada.

### LIME y SHAP

LIME y SHAP son metodos de explicabilidad relevantes en la literatura, pero no se incluyen como experimentos principales por alcance. En este TFM quedan como trabajo futuro o extension opcional.

## 13. Riesgos metodologicos y buenas practicas

### Data leakage

Ocurre cuando informacion del test aparece durante entrenamiento. En CT puede pasar si slices del mismo estudio se separan entre train y test. Se evita con split por `study_id`.

### Overfitting

El modelo memoriza train pero generaliza mal. Se controla con validacion, augmentation, early stopping y regularizacion.

### Sesgo de dataset

El modelo puede aprender caracteristicas del dataset en lugar de patologia. En COVID CXR se ha mostrado que algunos modelos aprenden atajos visuales. Por eso se necesita XAI y evaluacion cuidadosa.

### Interpretacion honesta de resultados negativos

No todo resultado bajo es un fracaso. En este TFM:

- CT clasificacion es dificil;
- CT segmentacion mejora con experimentacion pero queda limitada;
- Grad-CAM CT no se alinea con lesion.

Estos resultados sostienen una discusion realista sobre limites del enfoque 2D y datos anotados.

## 14. Resultados tecnicos que deben aparecer en la memoria

### Clasificacion

- CXR: `cxr_densenet121_weighted_ce`
  - accuracy `0.9496`
  - F1-macro `0.9567`
  - AUC macro `0.9931`
- CT:
  - mejor accuracy: `ct_resnet50_baseline`, accuracy `0.6484`
  - mejor F1/AUC: `ct_densenet121_baseline`, F1-macro `0.4173`, AUC `0.7229`

### Segmentacion

- CXR: `cxr_attention_unet_segmentation`, Dice `0.9853`, IoU `0.9715`
- CT: `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`, Dice `0.5637`, IoU `0.4305`

### XAI

- CXR Grad-CAM vs pulmon: IoU `0.2255`, ratio dentro mascara `0.3143`
- CT Grad-CAM vs lesion: IoU maximo observado `0.0146`, pico dentro mascara `0.0000`

## 15. Bibliografia base recomendada

### Datasets e imagen medica COVID

1. Morozov et al. (2020). MosMedData: Chest CT Scans with COVID-19 Related Findings. https://arxiv.org/abs/2005.06465
2. Rahman et al. COVID-19 Radiography Database. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
3. Rahman et al. (2020/2021). Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-rays Images. https://arxiv.org/abs/2012.02238
4. Tahir et al. (2021). COVID-19 Infection Localization and Severity Grading from Chest X-ray Images. https://arxiv.org/abs/2103.07985
5. DeGrave et al. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. https://www.nature.com/articles/s42256-021-00338-7
6. Rubin et al. (2020). Fleischner Society statement on chest imaging and COVID-19. https://pubs.rsna.org/doi/10.1148/radiol.2020201365

### Arquitecturas CNN y transfer learning

7. He et al. (2016). Deep Residual Learning for Image Recognition. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
8. Huang et al. (2017). Densely Connected Convolutional Networks. https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html
9. Tan and Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. https://proceedings.mlr.press/v97/tan19a.html
10. PyTorch Transfer Learning Tutorial. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
11. Torchvision Models Documentation. https://pytorch.org/vision/stable/models.html

### Segmentacion

12. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. https://arxiv.org/abs/1505.04597
13. Oktay et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. https://arxiv.org/abs/1804.03999
14. Salehi et al. (2017). Tversky loss function for image segmentation. https://arxiv.org/abs/1706.05721

### Losses, optimizacion y desbalanceo

15. Lin et al. (2017). Focal Loss for Dense Object Detection. https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
16. Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization / AdamW. https://arxiv.org/abs/1711.05101
17. PyTorch CrossEntropyLoss documentation. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

### Explicabilidad

18. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
19. Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. https://dl.acm.org/doi/10.1145/2939672.2939778
20. Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

### Metricas y herramientas

21. scikit-learn classification metrics documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
22. scikit-learn train_test_split documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
23. PyTorch data loading and `WeightedRandomSampler`. https://docs.pytorch.org/docs/stable/data.html
24. PyTorch `BCEWithLogitsLoss`. https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
25. PyTorch `AdamW`. https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
26. PyTorch `ReduceLROnPlateau`. https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

## 16. Frases utiles para redactar

- "La comparacion entre CXR y CT debe entenderse como comparacion metodologica entre modalidades y tareas, no como equivalencia clinica directa."
- "El uso de split por estudio en CT reduce el riesgo de fuga de informacion entre slices del mismo paciente."
- "El F1-macro es especialmente relevante en presencia de desbalance, porque otorga el mismo peso a cada clase."
- "Pixel accuracy en segmentacion puede ser enganosa cuando la region positiva ocupa una fraccion pequena de la imagen."
- "Grad-CAM proporciona una localizacion aproximada de evidencia visual, pero no equivale a una mascara de lesion."
- "La baja alineacion Grad-CAM vs lesion CT sugiere que la decision del clasificador no queda suficientemente respaldada por regiones patologicas anotadas."
- "Los resultados negativos en CT son informativos porque delimitan el alcance del enfoque 2D y de los datos anotados disponibles."
