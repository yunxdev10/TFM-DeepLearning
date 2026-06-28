# Borrador de introduccion del TFM

Fecha: 2026-05-20

## Datos base que conviene mencionar

### Dataset 1: CXR

El primer dataset es la **COVID-19 Radiography Database**, un conjunto publico de radiografias de torax usado ampliamente en trabajos de deteccion de COVID-19 mediante aprendizaje profundo.

Datos principales:

- Modalidad: radiografia de torax, CXR.
- Formato: imagenes 2D.
- Clases usadas en el TFM:
  - COVID-19: 3.616 imagenes.
  - Normal: 10.192 imagenes.
  - Lung Opacity: 6.012 imagenes.
  - Viral Pneumonia: 1.345 imagenes.
- Total: 21.165 imagenes.
- Incluye mascaras pulmonares asociadas.

Referencias utiles:

- COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Rahman et al. (2021), image enhancement y CXR: https://arxiv.org/abs/2012.02238
- Tahir et al. (2021), localizacion y severidad en CXR: https://arxiv.org/abs/2103.07985

### Dataset 2: CT

El segundo dataset es **MosMedData**, un conjunto de estudios de tomografia computarizada de torax con hallazgos relacionados con COVID-19.

Datos principales:

- Modalidad: tomografia computarizada, CT.
- Formato original: volumenes medicos CT.
- Tamano: 1.110 estudios CT.
- Clases originales: CT-0, CT-1, CT-2, CT-3, CT-4.
- En el TFM: CT-3 y CT-4 se agrupan como `CT-3+`.
- Incluye un subconjunto de 50 estudios con mascaras de infeccion anotadas.

Referencias utiles:

- Morozov et al. (2020), MosMedData: https://arxiv.org/abs/2005.06465
- Version oficial/permanente del dataset: https://mosmed.ai/datasets/covid19_1110

### Papers de contexto para justificar el TFM

- Fleischner Society: papel de la imagen toracica en COVID-19: https://pubs.rsna.org/doi/10.1148/radiol.2020201365
- DeGrave et al. (2021): riesgo de shortcut learning en modelos COVID CXR: https://www.nature.com/articles/s42256-021-00338-7
- ResNet: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
- DenseNet: https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html
- EfficientNet: https://proceedings.mlr.press/v97/tan19a.html
- U-Net: https://arxiv.org/abs/1505.04597
- Attention U-Net: https://arxiv.org/abs/1804.03999
- Grad-CAM: https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

## Introduccion propuesta

La pandemia de COVID-19 puso de manifiesto la importancia de disponer de herramientas de apoyo al diagnostico capaces de analizar informacion clinica de forma rapida, reproducible y escalable. Entre las fuentes de informacion utilizadas durante la crisis sanitaria, la imagen toracica tuvo un papel especialmente relevante, ya que la infeccion por SARS-CoV-2 puede producir alteraciones pulmonares visibles en radiografias de torax y tomografias computarizadas. Aunque las pruebas microbiologicas siguen siendo el elemento central para confirmar la enfermedad, sociedades radiologicas como la Fleischner Society destacaron que la imagen puede aportar informacion util en determinados escenarios clinicos, especialmente en la valoracion de afectacion pulmonar, gravedad y evolucion del paciente.

En este contexto, el aprendizaje profundo se ha consolidado como una de las aproximaciones mas prometedoras para el analisis automatico de imagen medica. Las redes neuronales convolucionales han demostrado una gran capacidad para extraer patrones visuales complejos a partir de imagenes, lo que ha favorecido su aplicacion en tareas como clasificacion diagnostica, segmentacion anatomica, deteccion de lesiones y apoyo a la interpretacion radiologica. Modelos como ResNet, DenseNet y EfficientNet han sido ampliamente utilizados como backbones de clasificacion mediante transfer learning, mientras que arquitecturas encoder-decoder como U-Net y Attention U-Net se han convertido en referencias para segmentacion medica.

Sin embargo, la aplicacion de deep learning a imagen medica presenta desafios metodologicos importantes. En primer lugar, los datasets clinicos suelen estar desbalanceados, tanto a nivel de clases como a nivel pixel a pixel en tareas de segmentacion. En segundo lugar, la disponibilidad de anotaciones precisas, como mascaras de lesion, es limitada debido al coste experto que requiere su generacion. En tercer lugar, la evaluacion basada exclusivamente en metricas globales puede ocultar errores relevantes en clases minoritarias o regiones patologicas pequenas. Finalmente, estudios recientes han mostrado que los modelos entrenados para detectar COVID-19 en radiografias pueden aprender atajos o correlaciones espurias relacionadas con la procedencia de los datos, el formato de imagen o caracteristicas no patologicas, en lugar de basarse en evidencias clinicamente relevantes. Por ello, no basta con obtener una alta accuracy: tambien es necesario analizar la robustez, la interpretabilidad y la coherencia anatomica de las predicciones.

Este Trabajo Fin de Master aborda estos retos mediante un estudio experimental sobre dos modalidades de imagen toracica: radiografia de torax y tomografia computarizada. Para la modalidad CXR se utiliza la COVID-19 Radiography Database, formada por 21.165 imagenes distribuidas en cuatro clases: COVID-19, Normal, Lung Opacity y Viral Pneumonia. Este dataset incluye ademas mascaras pulmonares, que permiten estudiar la segmentacion del campo pulmonar y analizar si las explicaciones visuales del modelo se concentran en regiones anatomicas plausibles. Para la modalidad CT se emplea MosMedData, un conjunto de 1.110 estudios de tomografia computarizada clasificados segun severidad radiologica, con un subconjunto de 50 estudios que incluye mascaras de infeccion. Esta segunda modalidad permite estudiar no solo la clasificacion de severidad, sino tambien la segmentacion de lesiones asociadas a COVID-19.

La eleccion de estos dos datasets permite comparar dos escenarios complementarios. La radiografia de torax es una modalidad 2D ampliamente disponible, con mayor volumen de datos y menor coste de adquisicion, pero con menor detalle anatomico. La tomografia computarizada, en cambio, proporciona informacion volumetrica mas rica y detallada, aunque con mayor coste, menor disponibilidad de anotaciones y mayor complejidad computacional. Por tanto, la comparacion entre CXR y CT en este trabajo no se plantea como una equivalencia clinica directa, sino como una comparacion metodologica entre dos tipos de imagen, dos estructuras de datos y dos niveles de dificultad.

El objetivo principal de este TFM es desarrollar y evaluar un pipeline reproducible de deep learning para clasificacion, segmentacion y explicabilidad en imagen toracica relacionada con COVID-19. Para la clasificacion se comparan distintas arquitecturas convolucionales preentrenadas, incluyendo ResNet-50, DenseNet-121 y EfficientNet-B0, junto con estrategias de tratamiento del desbalanceo como weighted cross-entropy, focal loss y oversampling. Para la segmentacion se entrenan modelos U-Net y Attention U-Net sobre mascaras pulmonares en CXR y mascaras de infeccion en CT, incorporando variantes especificas para mejorar la tarea CT, como Tversky loss, muestreo de parches positivos, contexto mixto, seleccion de umbral en validacion, modelos 2.5D y ensembles. Finalmente, se aplica Grad-CAM para estudiar la explicabilidad visual de los clasificadores y comparar la saliencia generada con las mascaras disponibles.

La contribucion del trabajo no se limita a entrenar modelos predictivos, sino que busca analizar de forma critica el comportamiento de dichos modelos. En particular, se evalua si las arquitecturas preentrenadas generalizan de forma diferente en CXR y CT, si las estrategias de balanceo mejoran realmente las clases minoritarias, si la segmentacion puede proporcionar informacion espacial fiable y si las explicaciones visuales se alinean con regiones anatomicas o patologicas relevantes. Esta perspectiva es especialmente importante en imagen medica, donde un resultado numericamente alto puede ser insuficiente si el modelo no se apoya en senales interpretables o clinicamente plausibles.

En conjunto, este TFM propone una evaluacion integrada de clasificacion, segmentacion y explicabilidad sobre dos datasets publicos de imagen toracica. El trabajo combina metricas cuantitativas, analisis cualitativo y discusion metodologica para ofrecer una vision realista de las posibilidades y limitaciones del deep learning aplicado a COVID-19. De este modo, se pretende aportar no solo una comparativa de modelos, sino tambien una reflexion sobre la importancia de la validacion rigurosa, el control del desbalanceo, la prevencion de fuga de datos y la interpretacion critica de los resultados en sistemas de inteligencia artificial aplicados a imagen medica.

## Version mas breve

La pandemia de COVID-19 impulso el desarrollo de sistemas de apoyo al diagnostico basados en inteligencia artificial, especialmente en el analisis de imagen toracica. Radiografias de torax y tomografias computarizadas han sido utilizadas para estudiar la afectacion pulmonar asociada a la enfermedad, aunque presentan diferencias importantes en disponibilidad, detalle anatomico, coste y tipo de informacion. En este contexto, el aprendizaje profundo ofrece herramientas potentes para clasificacion y segmentacion, pero tambien plantea retos relacionados con el desbalanceo de datos, la escasez de anotaciones, la interpretabilidad y el riesgo de aprendizaje de atajos no clinicos.

Este Trabajo Fin de Master analiza estos retos mediante dos datasets publicos: la COVID-19 Radiography Database, compuesta por 21.165 radiografias de torax distribuidas en cuatro clases, y MosMedData, formado por 1.110 estudios CT clasificados por severidad y un subconjunto de 50 estudios con mascaras de infeccion. Sobre estos datos se desarrolla un pipeline reproducible que compara arquitecturas convolucionales para clasificacion, modelos U-Net y Attention U-Net para segmentacion, estrategias de balanceo y tecnicas de explicabilidad visual mediante Grad-CAM.

El objetivo no es unicamente maximizar metricas, sino evaluar de forma critica en que medida los modelos aprenden patrones utiles, generalizan entre modalidades y producen explicaciones coherentes con las regiones anatomicas o patologicas disponibles. Asi, el trabajo integra clasificacion, segmentacion y explicabilidad para estudiar tanto el rendimiento como las limitaciones del deep learning aplicado a imagen medica relacionada con COVID-19.

## Ideas fuertes para defender en la introduccion

- No comparas solo dos modelos, sino dos modalidades medicas: CXR y CT.
- El trabajo integra tres tareas: clasificacion, segmentacion y explicabilidad.
- El uso de dos datasets publicos permite reproducibilidad.
- El desbalanceo y la escasez de mascaras son parte central del problema.
- Grad-CAM se justifica por el riesgo de shortcut learning.
- La segmentacion no es un adorno: sirve para evaluar localizacion y plausibilidad.
- La comparacion CXR vs CT debe interpretarse metodologicamente, no como equivalencia clinica directa.

## Citas recomendadas para colocar en la introduccion

Una combinacion equilibrada seria:

- Fleischner Society para justificar el papel clinico de la imagen toracica en COVID-19.
- COVID-19 Radiography Database para describir CXR.
- MosMedData para describir CT.
- DeGrave et al. para justificar explicabilidad y riesgo de shortcut learning.
- ResNet/DenseNet/EfficientNet para backbones de clasificacion.
- U-Net/Attention U-Net para segmentacion.
- Grad-CAM para explicabilidad visual.
