# Estado del arte - papers utiles para la memoria TFM

Fecha de cribado: 2026-05-12

## Enfoque recomendado para la redaccion

La memoria puede organizar el estado del arte en seis bloques:

1. Uso de imagen medica en COVID-19: CXR frente a CT, con cautela clinica.
2. Clasificacion con deep learning y transfer learning en CXR.
3. Clasificacion y severidad en CT.
4. Segmentacion pulmonar o de lesiones con U-Net y variantes con atencion.
5. Explicabilidad: Grad-CAM, SHAP, LIME y validacion visual.
6. Limitaciones metodologicas: desbalance, fuga de datos, shortcut learning y generalizacion.

Tu TFM encaja bien porque no se limita a reportar accuracy: compara CXR y CT con una metodologia equivalente, prueba estrategias de balanceo, separa clasificacion de segmentacion y plantea explicabilidad con una lectura anatomica distinta para cada modalidad.

---

## Papers prioritarios

### 1. COVID-Net: arquitectura especifica para deteccion COVID-19 en CXR

Referencia:
Wang, L., Lin, Z. Q. & Wong, A. (2020). "COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images". Scientific Reports.
URL: https://www.nature.com/articles/s41598-020-76550-z

Utilidad para tu TFM: alta.

Que aporta:
- Es uno de los trabajos mas citados sobre deteccion COVID-19 en radiografias.
- Defiende el uso de CXR como herramienta de cribado asistido.
- Introduce una arquitectura especifica, no solo transfer learning.
- Sirve para situar tu trabajo frente a modelos ad hoc.

Como usarlo en la memoria:
- En el estado del arte CXR, para mostrar que la literatura temprana propuso arquitecturas especificas para COVID-19.
- En la discusion, para justificar que tu enfoque es mas comparativo y reproducible: ResNet-50, DenseNet-121 y EfficientNet-B0 frente a redes disenadas ad hoc.

Limitacion a mencionar:
- Muchos trabajos iniciales de COVID-19 en CXR sufrieron problemas de generalizacion, tamano muestral o procedencia heterogenea de datos. Conviene no presentar estas accuracies como directamente comparables.

---

### 2. DeTraC: transfer learning y problemas de fronteras entre clases en CXR

Referencia:
Abbas, A., Abdelsamea, M. M. & Gaber, M. M. (2021). "Classification of COVID-19 in chest X-ray images using DeTraC deep convolutional neural network". Applied Intelligence.
URL: https://link.springer.com/article/10.1007/s10489-020-01829-7

Utilidad para tu TFM: alta-media.

Que aporta:
- Usa transfer learning y propone descomponer clases para manejar irregularidades en el dataset.
- Reporta accuracy alta en clasificacion COVID-19 con CXR.
- Es util para hablar de que el problema no es solo arquitectural, sino tambien de estructura de clases.

Como usarlo:
- Para justificar por que en tu TFM comparas estrategias de balanceo y no solo arquitecturas.
- Para reforzar que las fronteras COVID / neumonia / normal pueden ser ambiguas en CXR.

Limitacion:
- No es identico a tu planteamiento de 4 clases con Lung Opacity y Viral Pneumonia. Usarlo como contexto, no como benchmark directo.

---

### 3. Evaluacion de modelos preentrenados en CXR: DenseNet y transfer learning

Referencia:
Ucar, F. & Korkmaz, D. (2021). "Evaluation of deep learning-based approaches for COVID-19 classification based on chest X-ray images". Signal, Image and Video Processing.
URL: https://link.springer.com/article/10.1007/s11760-020-01820-2

Utilidad para tu TFM: alta.

Que aporta:
- Evalua varios modelos preentrenados en CXR.
- DenseNet121 obtiene muy buen rendimiento en clasificacion multiclase.
- Encaja directamente con tu resultado: DenseNet-121 con weighted CE ha sido el mejor CXR.

Como usarlo:
- En el apartado de clasificacion CXR con transfer learning.
- Como apoyo bibliografico para explicar por que DenseNet es una eleccion razonable: reutilizacion de caracteristicas, buen flujo de gradiente y eficiencia en problemas con datos limitados.

Limitacion:
- El dataset y protocolo no coinciden plenamente con el tuyo. Debes comparar tendencias, no numeros absolutos.

---

### 4. CXR con ResNet, DenseNet e Inception: comparacion de arquitecturas

Referencia:
Constantinou, M. et al. (2023). "COVID-19 Classification on Chest X-ray Images Using Deep Learning Methods". International Journal of Environmental Research and Public Health.
URL: https://www.mdpi.com/1660-4601/20/3/2035

Utilidad para tu TFM: alta.

Que aporta:
- Analiza modelos como ResNet50, ResNet101, DenseNet121, DenseNet169 e InceptionV3 con transfer learning.
- Se centra en CXR y evaluacion sobre datos no usados en entrenamiento/validacion.
- Es un antecedente claro para tu comparativa de arquitecturas.

Como usarlo:
- Para justificar la seleccion de ResNet-50 y DenseNet-121.
- Para redactar que la literatura suele apoyarse en backbones preentrenados por su rendimiento y coste razonable.

Limitacion:
- No incluye exactamente tu matriz con EfficientNet-B0 y estrategias de balanceo. Tu aportacion esta en la comparacion sistematica arquitectura x balanceo.

---

### 5. EfficientNetB0 y Grad-CAM en CXR

Referencia:
Zebin, T. & Rezvy, S. (2021). "COVID-19 detection and disease progression visualization: Deep learning on chest X-rays for classification and coarse localization". Applied Intelligence.
URL: https://link.springer.com/article/10.1007/s10489-020-01867-1

Utilidad para tu TFM: muy alta.

Que aporta:
- Usa VGG16, ResNet50 y EfficientNetB0 como extractores de caracteristicas.
- Reporta mejor rendimiento con EfficientNetB0 en su configuracion.
- Incorpora Grad-CAM para localizar regiones relevantes.
- Tambien aborda aumento de datos para la clase minoritaria COVID.

Como usarlo:
- Para justificar EfficientNet-B0 en tu matriz CXR.
- Para conectar clasificacion con explicabilidad visual.
- Para introducir la idea de que las saliencias ayudan a inspeccionar si el modelo mira regiones pulmonares plausibles.

Limitacion:
- La localizacion con Grad-CAM es gruesa y no equivale a una segmentacion clinica de lesion. Esto es importante para tu TFM: en CXR tus mascaras son pulmonares, no mascaras de lesion COVID.

---

### 6. COVID-19 Radiography Database / dataset base de CXR

Referencia de dataset:
COVID-19 Radiography Database. Kaggle.
URL: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Papers recomendados por el propio dataset:
- Chowdhury, M. E. H. et al. "Can AI help in screening Viral and COVID-19 pneumonia?" IEEE Access, 2020.
- Rahman, T. et al. "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images". Computers in Biology and Medicine, 2021.
URL: https://pubmed.ncbi.nlm.nih.gov/33799220/

Utilidad para tu TFM: imprescindible.

Que aporta:
- Es la fuente directa de tus datos CXR.
- Documenta el origen multiinstitucional y la evolucion del dataset.
- Incluye las clases usadas por tu TFM: COVID, Normal, Lung Opacity y Viral Pneumonia.
- Incluye mascaras pulmonares, relevantes para segmentacion pulmonar y para evaluar si XAI se concentra dentro del campo pulmonar.

Como usarlo:
- En la seccion de datasets.
- En metodologia, para describir composicion y procedencia.
- En limitaciones, para advertir que el dataset esta agregado a partir de varias fuentes, lo que puede introducir sesgos de adquisicion.

---

### 7. MosMedData: dataset CT con severidad COVID-19

Referencia:
Morozov, S. P. et al. (2020/2021). "MosMedData: data set of 1110 chest CT scans performed during the COVID-19 epidemic". Digital Diagnostics.
URL: https://jdigitaldiagnostics.com/DD/article/view/46826/en_US
Preprint: https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1.full-text

Utilidad para tu TFM: imprescindible.

Que aporta:
- Es la fuente central para tu parte CT.
- Describe 1110 estudios CT de pacientes de Moscu durante la pandemia.
- Incluye categorias CT-0 a CT-4 segun hallazgos/severidad.
- Contiene un subconjunto con mascaras de regiones de interes como opacidades en vidrio deslustrado y consolidaciones.

Como usarlo:
- En datasets y metodologia CT.
- Para justificar que CT no representa las mismas clases clinicas que CXR: en tu TFM CT es severidad/hallazgos, mientras CXR son categorias diagnosticas.
- Para justificar la fusion CT-3+ si CT-4 es escasa.

Limitacion:
- Las clases CT estan muy desbalanceadas y proceden de una cohorte geografica concreta. Esto refuerza la necesidad de split por estudio y metricas como F1-macro.

---

### 8. Clasificacion CT con muchas CNN y visualizacion

Referencia:
"Efficient and visualizable convolutional neural networks for COVID-19 classification using Chest CT". Expert Systems with Applications, 2022.
URL: https://www.sciencedirect.com/science/article/pii/S0957417422000392

Utilidad para tu TFM: alta.

Que aporta:
- Entrena y evalua 20 CNNs para clasificacion COVID-19 en CT.
- Reporta buen rendimiento de EfficientNetB5.
- Usa Grad-CAM para visualizar opacidades en vidrio deslustrado y consolidaciones.

Como usarlo:
- Para situar tu comparativa CT dentro de la tendencia de comparar multiples backbones.
- Para justificar que las visualizaciones en CT pueden contrastarse mejor con regiones patologicas que en CXR, cuando hay mascaras de lesion.

Limitacion:
- Es clasificacion COVID/no COVID en CT, no necesariamente severidad CT-0/CT-3+ como en tu caso. Usarlo para metodologia y XAI, no como comparacion directa.

---

### 9. Detection and Severity Classification of COVID-19 in CT Images

Referencia:
"Detection and Severity Classification of COVID-19 in CT Images Using Deep Learning". Diagnostics, 2021.
URL: https://www.mdpi.com/2075-4418/11/5/893

Utilidad para tu TFM: alta.

Que aporta:
- Trata deteccion y severidad en CT.
- Menciona MosMedData como validacion externa y describe clases de severidad.
- Es util para conectar CT con severidad, no solo diagnostico binario.

Como usarlo:
- En el apartado de CT, para justificar que CT se presta a tareas de cuantificacion/severidad.
- Para explicar por que tu comparacion CXR vs CT debe presentarse como comparacion metodologica y no como equivalencia clinica directa.

Limitacion:
- Debes revisar si su protocolo exacto de clases y datos coincide con el tuyo antes de comparar metricas.

---

### 10. U-Net vs SegNet en segmentacion COVID-19 CT

Referencia:
"COVID-19 lung CT image segmentation using deep learning methods: U-Net versus SegNet". BMC Medical Imaging, 2020.
URL: https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-020-00529-5

Utilidad para tu TFM: alta.

Que aporta:
- Compara arquitecturas de segmentacion en CT.
- Situa la segmentacion como herramienta para cuantificar regiones infectadas.
- Es un paper claro para introducir U-Net como baseline razonable.

Como usarlo:
- En Fase 2, para justificar U-Net en segmentacion CT.
- Para explicar que la segmentacion no solo localiza, sino que permite cuantificar extension de lesiones.

Limitacion:
- No cubre CXR pulmonar como tu proyecto; es mas directamente relevante para CT lesion.

---

### 11. Attention U-Net para segmentacion de infeccion COVID-19 en CT

Referencia:
Zhou, T., Canu, S. & Ruan, S. (2021). "Automatic COVID-19 CT segmentation using U-Net integrated spatial and channel attention mechanism".
URL: https://pubmed.ncbi.nlm.nih.gov/33362345/

Utilidad para tu TFM: muy alta.

Que aporta:
- Integra mecanismos de atencion espacial y de canal en U-Net.
- La motivacion es que las lesiones COVID pueden ser pequenas, difusas y con bordes ambiguos.
- Encaja directamente con tu inclusion de Attention U-Net.

Como usarlo:
- Para justificar Attention U-Net frente a U-Net vanilla.
- Para explicar que la atencion ayuda a filtrar regiones irrelevantes y mejorar la sensibilidad a patrones patologicos.

Limitacion:
- Atencion no garantiza interpretabilidad clinica por si sola. Conviene validarla con Dice/IoU y analisis visual.

---

### 12. D2A U-Net: atencion dual y convoluciones dilatadas

Referencia:
"D2A U-Net: Automatic segmentation of COVID-19 CT slices based on dual attention and hybrid dilated convolution". PubMed.
URL: https://pubmed.ncbi.nlm.nih.gov/34146799/

Utilidad para tu TFM: media-alta.

Que aporta:
- Propone una variante mas avanzada de U-Net para lesiones COVID en CT.
- Usa atencion dual y convoluciones dilatadas para ampliar campo receptivo.
- Sirve para mostrar la evolucion desde U-Net simple hacia modelos con atencion.

Como usarlo:
- En estado del arte de segmentacion CT, como ejemplo de refinamiento arquitectural.
- No hace falta implementarlo; basta con mencionarlo para contextualizar Attention U-Net y ResUNet.

Limitacion:
- Puede ser demasiado especifico si tu memoria necesita ser concisa. No lo pondria entre los 5 papers principales.

---

### 13. Grad-CAM: metodo base de explicabilidad visual

Referencia:
Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization". ICCV.
URL: https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

Utilidad para tu TFM: imprescindible.

Que aporta:
- Es la referencia base para mapas de calor dependientes de clase en CNN.
- No requiere cambiar la arquitectura ni reentrenar.
- Es especialmente adecuada para tus modelos ResNet, DenseNet y EfficientNet.

Como usarlo:
- En metodologia XAI.
- Para explicar que Grad-CAM produce localizacion gruesa de regiones relevantes para la prediccion.
- En limitaciones, aclarar que no es una prueba causal ni una segmentacion precisa.

---

### 14. LIME: explicaciones locales modelo-agnosticas

Referencia:
Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust You? Explaining the Predictions of Any Classifier".
URL: https://arxiv.org/abs/1602.04938

Utilidad para tu TFM: alta.

Que aporta:
- Explica predicciones individuales aproximando localmente el modelo con un modelo interpretable.
- Es modelo-agnostico, por tanto complementa a Grad-CAM.
- En imagenes suele trabajar con superpixeles.

Como usarlo:
- En metodologia XAI, para justificar una segunda familia de explicaciones.
- En discusion, para comparar explicaciones basadas en gradiente frente a perturbaciones.

Limitacion:
- En imagen medica puede ser sensible a la segmentacion en superpixeles y a los parametros de perturbacion.

---

### 15. SHAP: interpretabilidad basada en valores de Shapley

Referencia:
Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions". NeurIPS.
URL: https://arxiv.org/abs/1705.07874

Utilidad para tu TFM: media-alta.

Que aporta:
- Marco teorico unificado para atribucion de importancia.
- Buen fundamento si incluyes SHAP en el bloque de explicabilidad.

Como usarlo:
- Como referencia conceptual de SHAP.
- Para decir que SHAP busca atribuciones consistentes basadas en teoria de juegos.

Limitacion:
- En imagenes profundas puede ser computacionalmente pesado y menos directo de interpretar que Grad-CAM. Si el tiempo aprieta, Grad-CAM deberia ser prioritario.

---

### 16. DeepCOVIDExplainer

Referencia:
Karim, M. R. et al. (2020). "DeepCOVIDExplainer: Explainable COVID-19 Diagnosis from Chest X-ray Images".
URL: https://research.vu.nl/en/publications/deepcovidexplainer-explainable-covid-19-diagnosis-from-chest-x-ra/

Utilidad para tu TFM: alta.

Que aporta:
- Aplica explicabilidad directamente a diagnostico COVID-19 en CXR.
- Combina redes profundas con Grad-CAM++ y LRP.
- Sirve para mostrar que la explicabilidad no es accesoria en imagen medica, sino una necesidad para confianza clinica.

Como usarlo:
- En estado del arte XAI aplicado a CXR.
- Para justificar que tu TFM mida/compare saliencia con mascaras pulmonares o de lesion.

Limitacion:
- No usarlo como prueba de que las explicaciones sean clinicamente suficientes. La XAI visual requiere validacion cuidadosa.

---

### 17. Review de XAI en imagen COVID-19

Referencia:
Fuhrman, J. D. et al. (2021). "A review of explainable and interpretable AI with applications in COVID-19 imaging". Medical Physics.
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8646613/

Utilidad para tu TFM: muy alta.

Que aporta:
- Revisa tecnicas XAI e interpretabilidad aplicadas a imagen COVID-19.
- Es util para redactar el marco conceptual: interpretabilidad, confianza, limitaciones y validacion.

Como usarlo:
- Como referencia paraguas del apartado XAI.
- Para introducir la necesidad de no evaluar solo rendimiento predictivo.

Limitacion:
- Al ser review, debe complementar papers tecnicos originales como Grad-CAM, LIME y SHAP.

---

### 18. Shortcut learning y sesgos en CXR

Referencia:
Trivedi, A. et al. (2022). "Deep learning models for COVID-19 chest x-ray classification: Preventing shortcut learning using feature disentanglement". PLOS One.
URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274098

Utilidad para tu TFM: muy alta para discusion critica.

Que aporta:
- Trata el problema de shortcut learning: el modelo aprende senales espurias del dataset en lugar de patologia.
- Es clave para una memoria academica madura.

Como usarlo:
- En limitaciones metodologicas.
- Para justificar tus decisiones de splits, XAI y evaluacion de atencion dentro/fuera del pulmon.
- Para advertir que accuracy alta no basta.

Limitacion:
- No invalida el uso de CXR; obliga a disenar y discutir mejor la metodologia.

---

### 19. Focal Loss para desbalance

Referencia:
Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection". ICCV.
URL: https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html

Utilidad para tu TFM: imprescindible si mantienes `focal_loss`.

Que aporta:
- Propone focal loss para reducir el peso de ejemplos faciles y concentrar aprendizaje en ejemplos dificiles.
- Aunque nace en deteccion de objetos, se ha reutilizado mucho en clasificacion desbalanceada.

Como usarlo:
- En metodologia de balanceo.
- Para explicar por que comparas `baseline`, `weighted_ce` y `focal_loss`.

Limitacion:
- No asumir que focal loss siempre mejora; tus resultados CXR muestran que `weighted_ce` puede funcionar mejor. Eso es una discusion valiosa.

---

## Ranking de lectura recomendado

Si hay poco tiempo, leer en este orden:

1. MosMedData dataset paper.
2. COVID-19 Radiography Database + papers asociados de Chowdhury/Rahman.
3. Review sistematica 2023 sobre DL en CT y CXR.
4. Constantinou et al. 2023 o Ucar & Korkmaz 2021 para transfer learning CXR.
5. Zebin & Rezvy 2021 para EfficientNet + Grad-CAM.
6. Zhou et al. 2021 Attention U-Net CT.
7. Grad-CAM original.
8. Trivedi et al. 2022 sobre shortcut learning.
9. Focal Loss original.

---

## Ideas redactables para la memoria

### Idea 1 - Comparacion CXR/CT con cautela clinica

La literatura muestra que tanto CXR como CT han sido usadas para deteccion asistida de COVID-19, pero no son modalidades clinicamente equivalentes. La CXR suele formularse como clasificacion diagnostica entre COVID-19, neumonia, normalidad u opacidades, mientras que CT permite representar mejor extension y severidad de afectacion pulmonar. Por ello, en este TFM la comparacion CXR-CT debe interpretarse como comparacion metodologica bajo arquitecturas comunes, no como equivalencia directa entre etiquetas.

### Idea 2 - Por que transfer learning

El transfer learning domina gran parte de los trabajos de COVID-19 en imagen medica porque los datasets anotados son limitados y costosos de generar. Backbones como ResNet, DenseNet y EfficientNet permiten reutilizar representaciones visuales aprendidas en grandes colecciones y adaptarlas a imagen medica mediante fine-tuning.

### Idea 3 - Por que balanceo

El desbalance de clases es un problema recurrente en datasets COVID-19, especialmente en clases graves o minoritarias. Por ello, evaluar solo accuracy puede favorecer a la clase dominante. El TFM incorpora F1-macro, weighted CE y focal loss para estudiar si las estrategias de entrenamiento mejoran el rendimiento en clases minoritarias.

### Idea 4 - Por que XAI

En imagen medica, la explicabilidad permite inspeccionar si el modelo fundamenta sus predicciones en regiones anatomicas plausibles. Sin embargo, los mapas Grad-CAM, SHAP o LIME no deben interpretarse como segmentaciones clinicas. En CXR, la superposicion con mascaras pulmonares solo evalua si la atencion cae dentro del campo pulmonar; en CT, si existen mascaras de lesion, la comparacion puede aproximarse mas a una evaluacion patologica.

### Idea 5 - Aportacion concreta del TFM

Frente a muchos trabajos que reportan una unica arquitectura o una unica modalidad, este TFM aporta una matriz experimental comparable entre arquitecturas, estrategias de balanceo y modalidades, junto con una fase posterior de segmentacion y explicabilidad. La aportacion no es proponer una arquitectura nueva, sino una evaluacion controlada, reproducible y criticamente interpretada.

---

## Papers que usaria con cuidado

- Papers que reportan accuracies cercanas al 99% sin validacion externa, sin split por paciente/estudio o sin discutir sesgos de origen.
- Estudios binarios COVID vs normal si tu comparacion principal es multiclase.
- Estudios donde CT y CXR se mezclan sin aclarar que las etiquetas no tienen el mismo significado clinico.
- Trabajos que usan Grad-CAM como si fuera prueba diagnostica/localizacion exacta de lesion.

Estos papers pueden citarse como antecedentes, pero no deben ser el nucleo argumental.
