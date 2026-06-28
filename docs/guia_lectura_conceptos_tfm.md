# Guia de lectura por concepto para el TFM

Fecha: 2026-05-20

## Objetivo

Este documento sirve para estudiar y redactar el Estado del Arte, la Metodologia y la Discusion. No es solo una lista de referencias: debajo de cada concepto se indica que leer, que idea tecnica extraer y como convertirlo en un parrafo defendible.

La recomendacion practica es usarlo asi:

1. Leer primero el "parrafo para entender".
2. Abrir una o dos lecturas prioritarias.
3. Redactar con tus propias palabras usando la seccion "como llevarlo a la memoria".
4. Reservar las lecturas complementarias para justificar decisiones metodologicas concretas.

## 1. COVID-19, CXR y CT

### Concepto

COVID-19 puede producir alteraciones pulmonares visibles en imagen medica. En este TFM se usan dos modalidades:

- CXR: radiografia de torax 2D, barata y disponible, pero con menor detalle anatomico.
- CT: tomografia computarizada, formada por volumenes 3D de slices, con mas detalle pero mayor coste y menos anotaciones.

### Lecturas prioritarias

- Fleischner Society, consenso sobre imagen de torax en COVID-19: https://pubs.rsna.org/doi/10.1148/radiol.2020201365
- RSNA resumen del consenso Fleischner: https://www.rsna.org/news/2020/april/fleischner-statement-covid-19
- DeGrave et al., shortcut learning en COVID CXR: https://www.nature.com/articles/s42256-021-00338-7

### Parrafo para entender

La imagen medica en COVID-19 no debe presentarse como sustituto universal de la prueba clinica o microbiologica, sino como fuente de informacion complementaria. CXR y CT tienen naturalezas distintas: la CXR resume el torax en una proyeccion 2D, mientras que la CT proporciona cortes volumetricos con mayor detalle. Por eso, comparar resultados entre CXR y CT exige cautela: no se comparan solo dos datasets, sino tambien dos modalidades, dos tipos de etiqueta y dos niveles de dificultad clinica.

### Como llevarlo a la memoria

Puedes redactar que el TFM evalua deep learning en dos escenarios de imagen toracica complementarios. En CXR se aborda una clasificacion diagnostica de cuatro clases, mientras que en CT se aborda una clasificacion de severidad y una segmentacion de infeccion. Esta diferencia justifica que las metricas no se interpreten como una comparacion clinica directa entre modalidades, sino como una comparacion metodologica de pipelines.

## 2. Datasets CXR y CT

### Concepto

Un dataset no es solo un conjunto de imagenes. Tambien incluye clases, origen, tipo de anotacion, sesgos posibles, formato de datos y limitaciones.

### Lecturas prioritarias

- COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Rahman et al., CXR e image enhancement: https://arxiv.org/abs/2012.02238
- Tahir et al., localizacion y severidad en CXR: https://arxiv.org/abs/2103.07985
- MosMedData: https://arxiv.org/abs/2005.06465

### Parrafo para entender

En CXR, el dataset aporta imagenes clasificadas y mascaras pulmonares. Esas mascaras no son mascaras de lesion COVID: indican region pulmonar. En CT, MosMedData aporta estudios por severidad y un subconjunto con mascaras de infeccion. Esta diferencia es esencial: en CXR la mascara permite medir plausibilidad anatomica, mientras que en CT permite comparar con una region patologica.

### Como llevarlo a la memoria

En la seccion de datos conviene separar tres elementos: modalidad, etiqueta y mascara. Para CXR se debe indicar que las clases representan categorias diagnosticas o radiologicas. Para CT se debe indicar que la etiqueta representa severidad y que las mascaras de infeccion solo estan disponibles para un subconjunto. Esto ayuda a explicar por que CXR obtiene resultados altos y CT es mas dificil.

## 3. Sesgo de dataset y shortcut learning

### Concepto

Un modelo puede aprender atajos visuales que correlacionan con la etiqueta, pero no con la patologia. En COVID CXR esto es especialmente importante porque muchos datasets combinan fuentes distintas.

### Lectura prioritaria

- DeGrave et al., AI for radiographic COVID-19 detection selects shortcuts over signal: https://www.nature.com/articles/s42256-021-00338-7

### Parrafo para entender

Shortcut learning ocurre cuando el modelo usa senales espurias, por ejemplo marcas, diferencias de adquisicion, texto, encuadre o procedencia del hospital, en lugar de patrones patologicos. Un resultado alto en accuracy no garantiza por si solo que el modelo este usando informacion clinicamente relevante. Por eso en este TFM se complementan las metricas con Grad-CAM y con comparacion frente a mascaras disponibles.

### Como llevarlo a la memoria

En la discusion, este concepto permite justificar por que se incluye explicabilidad. Tambien permite interpretar con prudencia los resultados altos de CXR: el rendimiento es bueno, pero debe analizarse si la saliencia cae en regiones anatomicas plausibles.

## 4. CT, Hounsfield Units, windowing y NIfTI

### Concepto

Las CT originales son volumenes medicos, normalmente en formatos como NIfTI. Sus intensidades tienen significado fisico aproximado en unidades Hounsfield. El windowing selecciona un rango de intensidades para visualizar o procesar tejidos concretos.

### Lecturas prioritarias

- MosMedData y formato CT del dataset: https://arxiv.org/abs/2005.06465
- Hounsfield Units, StatPearls/NCBI: https://www.ncbi.nlm.nih.gov/books/NBK547721/
- NIfTI documentation: https://nifti.nimh.nih.gov/nifti-1/documentation
- Practical Window Setting Optimization for Medical Image Deep Learning: https://arxiv.org/abs/1812.00572

### Parrafo para entender

En CT, los pixeles no son simples niveles de gris arbitrarios: representan atenuacion radiologica en una escala relacionada con unidades Hounsfield. Como el rango completo de HU es amplio, se aplica una ventana para resaltar estructuras de interes. En este TFM se usa una ventana de pulmon aproximada para convertir volumenes CT en slices 2D adecuados para redes convolucionales.

### Como llevarlo a la memoria

Puedes explicar que el preprocesamiento CT no es equivalente al de CXR. En CT se parte de volumenes NIfTI, se aplica windowing HU, se extraen slices centrales, se redimensionan y se guardan como imagenes 2D. Esta conversion hace viable el entrenamiento local, pero tambien implica perdida de contexto volumetrico.

## 5. Split, estratificacion y fuga de datos

### Concepto

El split train/validation/test define que datos sirven para entrenar, ajustar decisiones y evaluar. En CT es critico separar por estudio/paciente, no por slice.

### Lecturas prioritarias

- scikit-learn `train_test_split` con `stratify`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Survey on deep learning with class imbalance, para entender distribuciones sesgadas: https://link.springer.com/article/10.1186/s40537-019-0192-5

### Parrafo para entender

Un split incorrecto puede inflar artificialmente los resultados. En CT, slices consecutivos del mismo estudio son muy parecidos. Si unos slices del mismo paciente estan en train y otros en test, el modelo puede reconocer caracteristicas del estudio en lugar de generalizar a pacientes nuevos. Por eso el TFM usa split por `study_id`.

### Como llevarlo a la memoria

En metodologia hay que destacar que CXR usa split estratificado por clase y CT usa split por estudio. Esta decision es una fortaleza metodologica porque reduce data leakage. La validacion se usa para early stopping y seleccion de threshold; el test se reserva para evaluacion final.

## 6. Data augmentation

### Concepto

Data augmentation crea variaciones de entrenamiento para mejorar generalizacion. En imagen medica debe usarse con cuidado para no crear anatomia irreal.

### Lecturas prioritarias

- Shorten y Khoshgoftaar, survey de image augmentation: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
- Torchvision transforms: https://docs.pytorch.org/vision/stable/transforms.html

### Parrafo para entender

La augmentation introduce variabilidad artificial: giros pequenos, flips, cambios de intensidad o transformaciones geometricas suaves. En tareas medicas no toda transformacion valida en imagen natural es clinicamente razonable. Por eso se usan augmentations conservadoras, especialmente en CT.

### Como llevarlo a la memoria

Puedes redactar que la augmentation se empleo como regularizacion y aumento efectivo de diversidad, pero manteniendo transformaciones plausibles. Esto es importante en datasets limitados o desbalanceados.

## 7. CNN, feature extraction y transfer learning

### Concepto

Las CNN aprenden filtros espaciales que detectan patrones locales y progresivamente mas complejos. Transfer learning reutiliza modelos preentrenados en ImageNet para reducir la necesidad de datos medicos masivos.

### Lecturas prioritarias

- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Torchvision modelos preentrenados: https://pytorch.org/vision/stable/models.html
- ImageNet deep CNN historico, AlexNet: https://dl.acm.org/doi/10.1145/3065386

### Parrafo para entender

Una CNN no recibe instrucciones explicitas sobre que textura o forma buscar. Aprende filtros que transforman pixeles en representaciones. Con transfer learning, el modelo parte de pesos aprendidos en un conjunto grande de imagenes naturales. Aunque ImageNet no es imagen medica, esas representaciones iniciales suelen ser utiles para bordes, formas y texturas.

### Como llevarlo a la memoria

El TFM puede justificar el uso de modelos preentrenados porque los datasets medicos disponibles son limitados. La estrategia de dos fases, primero cabeza y despues fine-tuning, adapta las representaciones al dominio medico evitando modificar todo el backbone desde el inicio.

## 8. ResNet-50

### Concepto

ResNet usa conexiones residuales para facilitar el entrenamiento de redes profundas.

### Lectura prioritaria

- He et al., Deep Residual Learning for Image Recognition: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

### Parrafo para entender

El problema de redes muy profundas es que no siempre mejoran al anadir capas: pueden sufrir degradacion y dificultad de optimizacion. ResNet introduce skip connections, que permiten aprender una correccion residual respecto a la entrada de un bloque. Esto estabiliza el flujo de gradiente y hace que redes profundas sean entrenables.

### Como llevarlo a la memoria

Puedes presentar ResNet-50 como baseline robusto y ampliamente validado. En los resultados CT tuvo la mejor accuracy, lo que indica que su arquitectura fue competitiva aunque no dominara todas las metricas.

## 9. DenseNet-121

### Concepto

DenseNet conecta capas de forma densa, reutilizando caracteristicas de etapas anteriores.

### Lectura prioritaria

- Huang et al., Densely Connected Convolutional Networks: https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html

### Parrafo para entender

DenseNet favorece el flujo de informacion porque cada capa recibe como entrada las caracteristicas producidas por capas previas. Esto reduce redundancia, mejora el flujo de gradiente y puede ser util en imagen medica, donde texturas y patrones de diferentes escalas son relevantes.

### Como llevarlo a la memoria

En tu TFM DenseNet-121 debe aparecer como arquitectura principal en CXR, porque con weighted cross-entropy produjo el mejor resultado global de clasificacion. En CT tambien fue importante porque obtuvo mejor F1/AUC que otros modelos, aunque ResNet tuviera mayor accuracy.

## 10. EfficientNet-B0

### Concepto

EfficientNet propone escalar profundidad, anchura y resolucion de forma equilibrada.

### Lectura prioritaria

- Tan y Le, EfficientNet: https://proceedings.mlr.press/v97/tan19a.html

### Parrafo para entender

EfficientNet parte de la idea de que aumentar solo profundidad, solo anchura o solo resolucion no siempre es eficiente. El compound scaling busca una proporcion equilibrada entre esos tres ejes. EfficientNet-B0 es la version base, ligera y eficiente.

### Como llevarlo a la memoria

Puedes incluir EfficientNet-B0 como arquitectura compacta dentro de la comparativa. En los resultados fue competitivo, pero no el mejor final, lo que permite explicar que eficiencia arquitectonica no garantiza superioridad en todos los dominios medicos.

## 11. Desbalanceo de clases

### Concepto

El desbalance ocurre cuando unas clases tienen muchas mas muestras que otras. En medicina, las clases minoritarias suelen ser precisamente las mas importantes.

### Lecturas prioritarias

- Johnson y Khoshgoftaar, Survey on deep learning with class imbalance: https://link.springer.com/article/10.1186/s40537-019-0192-5
- Focal Loss, Lin et al.: https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- PyTorch `WeightedRandomSampler`: https://docs.pytorch.org/docs/stable/data.html
- PyTorch `CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

### Parrafo para entender

En un dataset desbalanceado, accuracy puede ser enganosa: un modelo puede acertar muchas muestras mayoritarias y fallar clases minoritarias. Por eso el TFM compara baseline, weighted CE, focal loss y oversampling, y reporta F1-macro, matriz de confusion y AUC macro.

### Como llevarlo a la memoria

La metodologia de balanceo puede redactarse como una pregunta experimental: no se presupone que una tecnica sea mejor, sino que se compara su efecto. Weighted CE modifica la perdida; oversampling modifica la distribucion de batches; focal loss reduce la contribucion de ejemplos faciles y enfatiza los dificiles.

## 12. Weighted CE, Focal Loss y Oversampling

### Concepto

Son tres estrategias diferentes para tratar el desbalance.

### Lecturas prioritarias

- Focal Loss, paper fundacional: https://arxiv.org/abs/1708.02002
- PyTorch `WeightedRandomSampler`: https://docs.pytorch.org/docs/stable/data.html
- PyTorch `CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

### Parrafo para entender

Weighted CE aumenta el coste de equivocarse en clases minoritarias. Focal loss reduce el impacto de ejemplos faciles y concentra el aprendizaje en casos dificiles. Oversampling repite muestras minoritarias con mayor probabilidad. Estas tecnicas pueden ayudar, pero tambien pueden empeorar si aumentan sobreajuste o si el problema no esta limitado solo por frecuencia de clases.

### Como llevarlo a la memoria

En resultados puedes decir que weighted CE fue beneficiosa en CXR, mientras que en CT no resolvio claramente el problema. Esto es defendible: CT no solo tiene desbalance, sino tambien mayor dificultad visual, etiquetas de severidad y variabilidad entre slices.

## 13. Logits, softmax, sigmoid y funciones de perdida

### Concepto

El modelo produce logits. Para clasificacion multiclase se usa softmax; para segmentacion binaria se usa sigmoid por pixel.

### Lecturas prioritarias

- PyTorch `CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch `BCEWithLogitsLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

### Parrafo para entender

Los logits son salidas no normalizadas. En clasificacion, `CrossEntropyLoss` combina internamente softmax y perdida logaritmica. En segmentacion binaria, `BCEWithLogitsLoss` combina sigmoid y BCE de forma numericamente estable. Por eso no se debe aplicar sigmoid/softmax manualmente antes de estas losses cuando la funcion ya lo incorpora.

### Como llevarlo a la memoria

Puedes explicar que clasificacion y segmentacion usan salidas distintas: una probabilidad por clase frente a una probabilidad por pixel. Esta diferencia justifica el uso de losses distintas.

## 14. Optimizacion: AdamW, weight decay, learning rate y scheduler

### Concepto

La optimizacion define como se actualizan los pesos del modelo.

### Lecturas prioritarias

- AdamW, Loshchilov y Hutter: https://arxiv.org/abs/1711.05101
- PyTorch `AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch `ReduceLROnPlateau`: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

### Parrafo para entender

AdamW es una variante de Adam que desacopla weight decay de la actualizacion basada en gradientes. El learning rate controla el tamano de los pasos de optimizacion. ReduceLROnPlateau reduce el learning rate cuando la validacion deja de mejorar, permitiendo ajustes mas finos al final del entrenamiento.

### Como llevarlo a la memoria

En metodologia puedes indicar que se uso AdamW con weight decay como regularizacion, y scheduler basado en estancamiento para mejorar estabilidad. No hace falta desarrollar toda la matematica, pero si explicar el papel de cada hiperparametro.

## 15. Early stopping, validacion y test

### Concepto

Early stopping detiene el entrenamiento cuando la validacion deja de mejorar. Validation no es test.

### Lecturas prioritarias

- PyTorch training/fine-tuning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html

### Parrafo para entender

El conjunto de validacion sirve para tomar decisiones durante el desarrollo: detener entrenamiento, seleccionar thresholds o escoger hiperparametros. El test debe mantenerse como evaluacion final. Si se ajusta el modelo mirando test repetidamente, las metricas dejan de ser una estimacion honesta de generalizacion.

### Como llevarlo a la memoria

Destaca que thresholds y pesos de ensemble se seleccionaron en validacion. Esta frase es importante para defender buena practica experimental.

## 16. Metricas de clasificacion

### Concepto

No basta con accuracy. En multiclase y con desbalance hay que mirar precision, recall, F1-macro, F1-weighted, AUC y matriz de confusion.

### Lecturas prioritarias

- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- multiclass ROC example: https://sklearn.org/stable/auto_examples/model_selection/plot_roc.html

### Parrafo para entender

Accuracy mide aciertos globales, pero no dice si las clases minoritarias se detectan bien. F1-macro da el mismo peso a cada clase y por eso es mas informativo en datasets desbalanceados. La matriz de confusion permite ver errores especificos entre clases.

### Como llevarlo a la memoria

En resultados CT, usa F1-macro para explicar por que el mejor modelo por accuracy no siempre es el mas equilibrado. En CXR, la combinacion de accuracy, F1-macro y AUC alto apoya una conclusion mas fuerte.

## 17. Bootstrap y McNemar

### Concepto

Bootstrap estima incertidumbre mediante remuestreo. McNemar compara dos clasificadores sobre los mismos ejemplos usando discordancias.

### Lecturas prioritarias

- Dietterich, comparacion estadistica de clasificadores: https://colab.ws/articles/10.1162%2F089976698300017197
- McNemar original, referencia historica: https://ideas.repec.org/a/spr/psycho/v12y1947i2p153-157.html
- BMC Medical Research Methodology sobre variantes de McNemar: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-91

### Parrafo para entender

Dos modelos evaluados sobre el mismo test no son independientes. McNemar se centra en los casos donde uno acierta y el otro falla. Si las discordancias estan equilibradas, no hay evidencia clara de que uno sea mejor. Bootstrap, en cambio, permite estimar intervalos de confianza para metricas como accuracy o F1.

### Como llevarlo a la memoria

En Fase 4 puedes decir que las comparaciones top-2 no fueron estadisticamente concluyentes. Eso evita sobreafirmar diferencias pequenas.

## 18. Segmentacion semantica

### Concepto

La segmentacion asigna una etiqueta a cada pixel. En CXR se segmenta pulmon; en CT se segmenta lesion/infeccion.

### Lecturas prioritarias

- U-Net: https://arxiv.org/abs/1505.04597
- Attention U-Net: https://arxiv.org/abs/1804.03999
- Metrics for evaluating 3D medical image segmentation: https://link.springer.com/article/10.1186/s12880-015-0068-x

### Parrafo para entender

La clasificacion responde que clase tiene una imagen. La segmentacion responde donde esta una estructura. Por eso es una tarea mas exigente: requiere coherencia espacial y comparacion pixel a pixel con una mascara.

### Como llevarlo a la memoria

En el TFM, la segmentacion cumple dos funciones: resultado propio y soporte para interpretabilidad. Las mascaras permiten evaluar no solo Dice/IoU, sino tambien si Grad-CAM cae en zonas plausibles.

## 19. U-Net y Attention U-Net

### Concepto

U-Net es una red encoder-decoder con skip connections. Attention U-Net anade mecanismos de atencion en las conexiones para filtrar informacion irrelevante.

### Lecturas prioritarias

- U-Net, Ronneberger et al.: https://arxiv.org/abs/1505.04597
- Attention U-Net, Oktay et al.: https://arxiv.org/abs/1804.03999

### Parrafo para entender

El encoder comprime informacion y captura contexto; el decoder reconstruye resolucion espacial. Las skip connections recuperan detalle perdido. Attention U-Net intenta que esas conexiones no pasen toda la informacion sin filtro, sino que ponderen regiones utiles para la tarea.

### Como llevarlo a la memoria

Puedes explicar que Attention U-Net fue razonable para imagen medica porque combina localizacion fina y contexto. En resultados, fue la mejor familia tanto en CXR como en CT.

## 20. Dice, IoU, pixel accuracy y desbalance pixel a pixel

### Concepto

Dice e IoU miden solapamiento. Pixel accuracy puede ser enganosa cuando la region positiva es pequena.

### Lecturas prioritarias

- Taha y Hanbury, metricas de segmentacion medica: https://link.springer.com/article/10.1186/s12880-015-0068-x
- Dice/Jaccard optimization in medical segmentation: https://arxiv.org/abs/2010.13499
- Generalised Dice loss para segmentacion desbalanceada: https://arxiv.org/abs/1707.03237

### Parrafo para entender

En segmentacion CT, la lesion suele ocupar pocos pixeles. Un modelo que predice casi todo como fondo puede tener pixel accuracy alta, pero Dice bajo. Dice e IoU se centran en el solapamiento de la region positiva, por eso son mas adecuados para evaluar lesion.

### Como llevarlo a la memoria

Cuando expliques resultados CT, no uses pixel accuracy como metrica principal. Usala como secundaria y deja claro que Dice/IoU son las metricas mas informativas.

## 21. Dice loss, Tversky loss y `pos_weight`

### Concepto

Estas losses intentan combatir el desbalance entre fondo y lesion.

### Lecturas prioritarias

- Tversky loss: https://arxiv.org/abs/1706.05721
- Generalised Dice loss: https://arxiv.org/abs/1707.03237
- PyTorch `BCEWithLogitsLoss` y `pos_weight`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

### Parrafo para entender

BCE trata cada pixel como una clasificacion binaria. Dice loss optimiza solapamiento. Tversky permite ponderar falsos positivos y falsos negativos, algo util cuando la lesion es pequena y el coste de no detectarla no es igual que el de sobresegmentar.

### Como llevarlo a la memoria

Puedes explicar que las variantes CT se orientaron a reducir el desbalance pixel a pixel. La combinacion Tversky+BCE busca mantener aprendizaje pixel a pixel y mejorar el solapamiento.

## 22. Threshold, postprocesado y componentes conectados

### Concepto

El modelo de segmentacion produce probabilidades. Un threshold las convierte en mascara binaria. El postprocesado elimina ruido o componentes pequenas.

### Lecturas prioritarias

- Taha y Hanbury, discusion sobre threshold y metricas: https://link.springer.com/article/10.1186/s12880-015-0068-x
- PyTorch BCE/sigmoid como base probabilistica: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

### Parrafo para entender

El threshold no es un detalle menor. Cambiar de 0.5 a 0.8 o 0.9 puede reducir falsos positivos, pero tambien puede aumentar falsos negativos. Por eso debe elegirse en validacion, no en test. Las componentes conectadas permiten eliminar pequenas islas de prediccion, aunque en este TFM la mejora fue limitada.

### Como llevarlo a la memoria

Puedes redactar que se hizo analisis de postprocesado y busqueda de threshold para comprobar si el problema CT era solo de decision final. Como la mejora fue pequena, la conclusion es que la limitacion era mas estructural: datos, contexto y capacidad.

## 23. Patch-based training y positive crop sampling

### Concepto

El entrenamiento por patches recorta zonas de la imagen para aumentar detalle local. Positive crop sampling fuerza que muchos crops contengan lesion.

### Lecturas prioritarias

- U-Net, uso historico de patches y segmentacion con pocos datos: https://arxiv.org/abs/1505.04597
- Generalised Dice loss y segmentacion desbalanceada: https://arxiv.org/abs/1707.03237
- Metrics/thresholding en segmentacion: https://link.springer.com/article/10.1186/s12880-015-0068-x

### Parrafo para entender

En CT hay muchas regiones de fondo y pocas de lesion. Si se entrena con crops aleatorios, gran parte de los batches pueden no aportar informacion positiva. Centrar crops en pixeles de lesion incrementa la exposicion del modelo a la clase minoritaria, pero si se exagera puede reducir contexto anatomico.

### Como llevarlo a la memoria

Esta parte es clave para defender experimentacion: se probo patch puro, mixed context y diferentes probabilidades de crop positivo. El mejor resultado vino de equilibrar foco local y contexto global.

## 24. Mixed context, 2.5D y slices negativos

### Concepto

Mixed context combina detalle local y contexto global. 2.5D usa slices vecinos como canales. Slices negativos ensenan al modelo cuando no debe segmentar.

### Lecturas prioritarias

- Flexible 2.5D medical image segmentation: https://arxiv.org/abs/2405.00130
- CSAM 2.5D cross-slice attention: https://arxiv.org/abs/2311.04942
- 2.5D kidney/tumor segmentation: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02189-1

### Parrafo para entender

Un modelo 2D ve un slice aislado. Un modelo 3D completo ve el volumen, pero consume mas memoria y necesita mas datos. El enfoque 2.5D es un compromiso: apila slices vecinos para introducir contexto volumetrico sin entrenar una red 3D completa. En este TFM, 2.5D no supero al mejor 2D, lo cual tambien es informativo.

### Como llevarlo a la memoria

En discusion puedes decir que el contexto volumetrico fue explorado, pero que la mejor mejora vino de capacidad y estrategia de crops. Esto evita que parezca que no se penso en la naturaleza 3D de CT.

## 25. Capacidad del modelo, `base_features` y ablation study

### Concepto

La capacidad indica cuantos patrones puede representar el modelo. En U-Net se controla parcialmente con `base_features`. Un ablation study cambia un componente para medir su efecto.

### Lecturas prioritarias

- EfficientNet, relacion entre escala y rendimiento: https://proceedings.mlr.press/v97/tan19a.html
- U-Net como base encoder-decoder: https://arxiv.org/abs/1505.04597
- 2.5D empirical discussion como ejemplo de comparativa metodologica: https://www.sciencedirect.com/science/article/pii/S0895611122000611

### Parrafo para entender

Aumentar capacidad puede mejorar el aprendizaje, pero tambien aumenta riesgo de sobreajuste. Por eso debe evaluarse empiricamente. En el TFM, la variante `bf32` aumento `base_features` y obtuvo el mejor resultado CT, indicando que la receta previa era prometedora pero estaba limitada por capacidad.

### Como llevarlo a la memoria

Puedes presentar la secuencia experimental CT como ablacion: baseline, Tversky, patch, mixed context, 2.5D, ensemble y aumento de capacidad. Esta narrativa es fuerte porque muestra aprendizaje experimental progresivo.

## 26. Ensemble por promedio de probabilidades

### Concepto

Un ensemble combina modelos para aprovechar errores complementarios. En segmentacion se pueden promediar probabilidades pixel a pixel.

### Lecturas prioritarias

- Knowledge distillation with ensembles for medical image segmentation: https://pmc.ncbi.nlm.nih.gov/articles/PMC9142841/
- Deep ensembles, Lakshminarayanan et al.: https://arxiv.org/abs/1612.01474

### Parrafo para entender

Promediar probabilidades suaviza decisiones individuales y puede mejorar estabilidad si los modelos cometen errores distintos. El peso del ensemble debe elegirse en validacion. En este TFM, el ensemble mejoro el mejor modelo individual previo, pero despues fue superado por `bf32`.

### Como llevarlo a la memoria

Presenta el ensemble como experimento intermedio y no como resultado final. Esto demuestra que se exploraron combinaciones, pero que la seleccion final se baso en evidencia.

## 27. Grad-CAM y explicabilidad

### Concepto

Grad-CAM genera mapas de calor usando gradientes de la clase objetivo respecto a activaciones convolucionales.

### Lecturas prioritarias

- Grad-CAM, paper ICCV: https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Grad-CAM arXiv: https://arxiv.org/abs/1610.02391
- DeGrave et al. como motivacion de XAI en COVID CXR: https://www.nature.com/articles/s42256-021-00338-7

### Parrafo para entender

Grad-CAM no produce una mascara clinica. Produce una localizacion aproximada de regiones que contribuyen a la prediccion. En CNNs medicas es util para auditar plausibilidad: si el mapa se concentra fuera del pulmon o fuera de la lesion disponible, hay razones para desconfiar o al menos matizar el resultado.

### Como llevarlo a la memoria

En CXR, Grad-CAM frente a mascara pulmonar se interpreta como plausibilidad anatomica, no localizacion de lesion. En CT, Grad-CAM frente a mascara de infeccion se interpreta como alineacion patologica. Esta distincion es esencial.

## 28. LIME y SHAP

### Concepto

LIME y SHAP son metodos de explicabilidad relevantes, pero en este TFM se dejan como trabajo futuro por alcance.

### Lecturas prioritarias

- LIME: https://arxiv.org/abs/1602.04938
- SHAP: https://papers.nips.cc/paper/7062-a-unified-approach-to-int

### Parrafo para entender

LIME aproxima localmente el modelo con una explicacion interpretable. SHAP asigna importancia a caracteristicas usando una base teorica inspirada en valores de Shapley. Ambos son utiles, pero en imagen medica pueden requerir decisiones adicionales, como segmentacion en superpixeles o alto coste computacional.

### Como llevarlo a la memoria

Puedes mencionarlos en Estado del Arte y justificar que el experimento principal se centra en Grad-CAM porque es natural para CNNs y suficiente para analizar alineacion visual con mascaras disponibles.

## 29. Interpretacion de resultados negativos

### Concepto

Un resultado bajo no invalida el TFM si se interpreta con rigor. Puede revelar limites de datos, modelo o metodologia.

### Lecturas prioritarias

- DeGrave et al., limites de modelos COVID CXR: https://www.nature.com/articles/s42256-021-00338-7
- Taha y Hanbury, sensibilidad de metricas de segmentacion: https://link.springer.com/article/10.1186/s12880-015-0068-x
- Class imbalance survey: https://link.springer.com/article/10.1186/s40537-019-0192-5

### Parrafo para entender

En investigacion aplicada, no todo experimento debe mejorar el resultado anterior. Un experimento puede ser valioso si descarta una hipotesis. En este TFM, 2.5D, focal loss o postprocesado no fueron siempre superiores, pero ayudan a explicar que el problema CT no se arregla solo con un hiperparametro.

### Como llevarlo a la memoria

La discusion debe presentar los resultados CT como evidencia realista: el mejor modelo mejora respecto a baseline, pero la tarea sigue limitada por datos anotados, pequeno tamano de lesion, perdida de contexto 3D y dificultad de severidad.

## 30. Lectura minima recomendada antes de redactar

Si tienes poco tiempo, lee en este orden:

1. DeGrave et al. para entender sesgos y por que XAI importa.
2. MosMedData para entender CT, severidad y mascaras.
3. COVID-19 Radiography Database para describir CXR.
4. ResNet, DenseNet y EfficientNet solo en introduccion/metodologia.
5. U-Net y Attention U-Net para segmentacion.
6. Focal Loss y class imbalance survey para balanceo.
7. Grad-CAM para explicabilidad.
8. Taha y Hanbury para Dice, IoU y limites de metricas.

## 31. Frases puente utiles

- "La eleccion de F1-macro se justifica por la presencia de clases desbalanceadas y por la necesidad de no ocultar el rendimiento de clases minoritarias."
- "La seleccion de threshold se realizo en validacion para evitar ajuste directo sobre el conjunto de test."
- "La mascara CXR se interpreta como region anatomica pulmonar, no como anotacion de lesion."
- "La baja alineacion Grad-CAM en CT no invalida el clasificador, pero limita su interpretabilidad patologica."
- "La secuencia de experimentos CT constituye una ablacion progresiva sobre perdida, muestreo, contexto, capacidad y combinacion de modelos."
- "El uso de slices 2D reduce coste computacional, pero sacrifica contexto volumetrico completo."
- "El resultado negativo de una variante se conserva en la discusion porque informa sobre que modificaciones no fueron suficientes para mejorar la tarea."
