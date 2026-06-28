# Comparativa de trabajos relacionados y hueco que cubre este TFM

Fecha de preparacion: 2026-05-12

## Lectura estrategica

El tutor pide una tabla donde se vea que aporta el TFM frente a trabajos relacionados. La idea defendible no es decir que ningun trabajo haya hecho nada parecido, sino que **la mayoria de trabajos cubren solo una parte del problema**: CXR o CT, clasificacion o segmentacion, rendimiento o explicabilidad, una arquitectura o varias, una estrategia de balanceo o ninguna.

La contribucion diferenciadora de este TFM puede formularse asi:

> Este TFM realiza una comparacion experimental controlada de modelos de deep learning sobre dos modalidades de imagen medica, CXR y CT, usando una metodologia comun de clasificacion, balanceo, segmentacion y explicabilidad. A diferencia de muchos antecedentes, no se limita a optimizar una arquitectura o una unica modalidad, sino que integra rendimiento, desbalance, separacion por estudio en CT, segmentacion y XAI con una interpretacion critica de las diferencias clinicas entre modalidades.

## Criterios de comparacion

Columnas usadas en la tabla:

- **Modalidad**: CXR, CT o ambas.
- **Tarea**: clasificacion, segmentacion, severidad, explicabilidad.
- **Modelos**: arquitecturas principales usadas.
- **Balanceo**: si trata explicitamente desbalance de clases.
- **Segmentacion**: si incluye mascaras o modelos de segmentacion.
- **XAI**: si usa Grad-CAM, Grad-CAM++, LIME, SHAP u otro metodo.
- **Control metodologico**: split estratificado, validacion externa, split por paciente/estudio, cautela ante fuga de datos o shortcut learning.
- **Hueco frente a tu TFM**: que punto no cubre y que tu trabajo si puede cubrir.

---

## Tabla principal de comparacion

| Trabajo | Tipo | Modalidad / datos | Tarea principal | Modelos / enfoque | Balanceo | Segmentacion | XAI | Control metodologico | Hueco que deja | Como lo cubre tu TFM |
|---|---|---|---|---|---|---|---|---|---|---|
| Wang et al., 2020, **COVID-Net** ([Scientific Reports](https://www.nature.com/articles/s41598-020-76550-z)) | Articulo | CXR, COVIDx | Clasificacion CXR | Arquitectura propia COVID-Net | No es el eje | No | Visualizacion/explicabilidad limitada | Dataset publico COVIDx | Se centra en una arquitectura especifica y CXR; no compara CXR vs CT ni segmentacion/XAI sistematica | Tu TFM usa arquitecturas estandar comparables, incorpora CT y separa clasificacion, segmentacion y explicabilidad |
| Pham, 2021, **fine tuning vs modelos nuevos** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7680558/)) | Articulo | CXR, varias bases publicas | Clasificacion 2/3 clases | AlexNet, GoogLeNet, SqueezeNet | No central | No | No | Repite ejecuciones, pero mezcla fuentes | Buen antecedente de transfer learning, pero no aborda CT, balanceo profundo, segmentacion ni XAI | Tu TFM refuerza transfer learning con matriz 3 arquitecturas x 3 estrategias y dos modalidades |
| Ucar & Korkmaz / Kc et al., 2021, **evaluacion de modelos preentrenados** ([Springer](https://link.springer.com/article/10.1007/s11760-020-01820-2), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33432267/)) | Articulo | CXR, dataset pequeno multiclase | Clasificacion CXR | Ocho modelos preentrenados; DenseNet121 destacado | No principal | No | No | Dataset reducido | Demuestra potencia de DenseNet/transfer learning, pero no estudia CT ni estrategias de balanceo comparadas | Tu TFM usa DenseNet121 como uno de tres backbones y mide si weighted CE/focal loss ayudan |
| Zebin & Rezvy, 2021, **EfficientNetB0 + Grad-CAM** ([Springer](https://link.springer.com/article/10.1007/s10489-020-01867-1), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486976/)) | Articulo | CXR | Clasificacion y localizacion gruesa | VGG16, ResNet50, EfficientNetB0 | Aumento de datos | No como tarea de segmentacion | Grad-CAM | Uso de datasets publicos | Conecta clasificacion y XAI, pero solo CXR y sin CT/segmentacion formal | Tu TFM puede compararlo como antecedente XAI-CXR y ampliar a CT y mascaras |
| Constantinou et al., 2023, **clasificacion CXR con TL** ([MDPI](https://www.mdpi.com/1660-4601/20/3/2035), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36767399/)) | Articulo | CXR, COVID-QU-Ex / repositorio grande | Clasificacion CXR | ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3 | Dataset bastante equilibrado; no ablation de balanceo | No | No | Evalua en datos desconocidos no usados en train/val | Muy cercano para CXR, pero no incluye CT, segmentacion, XAI ni comparativa de balanceo | Tu TFM anade estrategias `baseline`, `weighted_ce`, `focal_loss`; incorpora CT y XAI |
| Rahman et al., 2021, **COVID-19 Radiography Database / enhancement + masks** ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S001048252100113X), [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)) | Articulo + dataset | CXR, COVID/Normal/Lung Opacity/Viral Pneumonia, mascaras pulmonares | Clasificacion, enhancement, segmentacion pulmonar | Varias CNN y U-Net modificada | No como matriz de estrategias | Si, pulmon | Visualizacion | Dataset grande con mascaras | Es base de datos clave, pero su objetivo es enhancement/segmentacion CXR; no compara CT ni severidad | Tu TFM reutiliza el dataset en una matriz comparativa y distingue mascaras pulmonares de lesiones reales |
| Trivedi et al., 2022, **shortcut learning en CXR** ([PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0274098), [Microsoft Research](https://www.microsoft.com/en-us/research/publication/deep-learning-models-for-covid-19-chest-x-ray-classification-preventing-shortcut-learning-using-feature-disentanglement/)) | Articulo critico | CXR, fuentes multiples | Generalizacion y sesgo | Feature disentanglement | No | No | Visualizaciones y test externo | Muy fuerte metodologicamente, pero no busca comparar CXR/CT ni segmentacion | Tu TFM puede apoyarse en el para justificar XAI, mascaras, cautela con accuracy y discusion de sesgos |
| Morozov et al., **MosMedData** ([medRxiv](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1.full-text)) | Dataset / data descriptor | CT, 1110 estudios, CT-0 a CT-4 | Dataset de severidad CT | No aplica | No aplica | Incluye subconjunto con anotaciones/mascaras | No | Etiquetado por severidad, datos por estudio | Es fuente de datos, no comparativa de modelos | Tu TFM aporta pipeline de clasificacion CT, split por `study_id`, fusion CT-3+ y comparacion con CXR |
| Qiblawey et al., 2021, **deteccion y severidad CT** ([MDPI](https://www.mdpi.com/2075-4418/11/5/893), [PubMed](https://pubmed.ncbi.nlm.nih.gov/34067937/)) | Articulo | CT, incluyendo MosMedData para severidad | Segmentacion pulmon/lesion, deteccion y severidad | U-Net/FPN con backbones ResNet/DenseNet | No como foco | Si, pulmon y lesion | No XAI clasico; localizacion por segmentacion | Validacion con MosMedData | Muy fuerte en CT, pero no integra CXR ni compara estrategias de balanceo de clasificadores 2D | Tu TFM conecta CT con CXR bajo metodologia comun y separa rendimiento de explicabilidad |
| Zhao et al., 2021, **deep learning CT detection** ([Scientific Reports](https://www.nature.com/articles/s41598-021-93832-2)) | Articulo | CT | Deteccion COVID en CT | CNN | No central | No | No | Dataset CT especifico | CT binario/deteccion, no severidad multiclase ni CXR | Tu TFM usa CT como severidad agrupada y lo compara metodologicamente con CXR |
| Yang et al., 2021, **CXR + CT en un mismo articulo** ([Scientific Reports](https://www.nature.com/articles/s41598-021-99015-3)) | Articulo | CXR y CT | Clasificacion COVID | VGG16, DenseNet121, ResNet50, ResNet152 | No como eje | No | No | Datasets publicos | Incluye ambas modalidades, pero no como matriz paralela con balanceo, segmentacion y XAI | Tu TFM tiene una lectura mas completa: balanceo, segmentacion y explicabilidad por modalidad |
| Saood & Hatem, 2021, **U-Net vs SegNet CT** ([BMC](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-020-00529-5), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7870362/)) | Articulo | CT | Segmentacion de regiones infectadas | U-Net, SegNet | No | Si, lesion CT | No | Dataset pequeno | Segmentacion CT pura; no clasificacion CXR/CT ni XAI | Tu TFM puede usarlo para justificar U-Net/Attention U-Net en CT, pero aporta tambien clasificacion y comparacion |
| Zhou et al., 2021, **U-Net con atencion espacial/canal** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/33362345/)) | Articulo | CT | Segmentacion COVID | U-Net + atencion + focal Tversky | Trata lesiones pequenas con loss | Si, lesion CT | Atencion interna, no XAI post-hoc | Dataset de 473 slices | Muy util para justificar Attention U-Net, pero no aborda CXR ni clasificacion multimodal | Tu TFM incorpora Attention U-Net como variante dentro de un marco mas amplio |
| Zhao et al., 2021, **D2A U-Net** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/34146799/), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010482521003206)) | Articulo | CT | Segmentacion lesion COVID | U-Net con doble atencion y convolucion dilatada | No como clasificacion | Si | Atencion interna | Evaluacion de segmentacion | Arquitectura avanzada de segmentacion CT, no clasificacion ni CXR | Tu TFM no necesita superar esta arquitectura; la usa para contextualizar por que la atencion es relevante |
| Karim et al., 2020, **DeepCOVIDExplainer** ([Elsevier profile](https://researchcollaborations.elsevier.com/en/publications/deepcovidexplainer-explainable-covid-19-diagnosis-from-chest-x-ra/), [VU](https://research.vu.nl/en/publications/deepcovidexplainer-explainable-covid-19-diagnosis-from-chest-x-ra/)) | Articulo/conferencia | CXR | Clasificacion explicable | Ensemble DNN | No central | No | Grad-CAM++, LRP | Dataset grande, CXR | Fuerte en XAI CXR, pero no CT ni mascaras de lesion/pulmon para evaluar saliencia | Tu TFM puede comparar Grad-CAM/SHAP/LIME y medir coherencia con mascaras segun modalidad |
| Selvaraju et al., 2017, **Grad-CAM** ([CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)) | Metodo base | General | Explicabilidad visual | Gradientes en ultima capa convolucional | No aplica | No | Si | No medico | Metodo fundacional, no COVID ni evaluacion clinica | Tu TFM lo aplica a CXR/CT y discute limites: mapa de calor no equivale a lesion |
| Ribeiro et al., 2016, **LIME** ([KDD](https://www.kdd.org/kdd2016/subtopic/view/why-should-i-trust-you-explaining-the-predictions-of-any-classifier), [CoLab DOI](https://colab.ws/articles/10.1145%2F2939672.2939778)) | Metodo base | General | Explicabilidad local modelo-agnostica | Modelo interpretable local | No aplica | No | Si | No medico | Metodo general, sensible a parametros/superpixeles en imagen medica | Tu TFM lo puede usar como contraste frente a Grad-CAM en ejemplos seleccionados |
| Lundberg & Lee, 2017, **SHAP** ([arXiv/HF](https://huggingface.co/papers/1705.07874)) | Metodo base | General | Importancia de variables / atribucion | Valores de Shapley | No aplica | No | Si | No medico | Metodo general y costoso para imagenes profundas | Tu TFM puede usarlo si el coste computacional es viable; si no, dejarlo como metodo discutido |
| Tzeng et al., 2023, **revision sistematica AI-CXR COVID** ([PubMed](https://pubmed.ncbi.nlm.nih.gov/36832072/)) | Revision sistematica | CXR | Diagnostico COVID asistido por IA | Multiples | Variable | Variable | Variable | Meta-analisis | Da panorama general, pero no es experimento propio | Sirve para justificar el estado del arte y la necesidad de cautela metodologica |
| Lee et al., 2023, **revision CT + X-ray COVID** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10218920/)) | Revision sistematica | CXR y CT | Interpretacion con deep learning | Multiples | Variable | Variable | Incluye explicabilidad | Revisa arquitectura, complejidad, disponibilidad de codigo/datos | Es paraguas teorico, no aporta pipeline nuevo | Tu TFM aterriza esas recomendaciones en una implementacion concreta |

---

## TFM / tesis relacionados localizados

| Trabajo | Tipo | Modalidad | Enfoque | Utilidad para la memoria | Diferencia frente a tu TFM |
|---|---|---|---|---|---|
| Garcia Astudillo, Victor, **Deteccion de COVID-19 en radiografias con Deep Learning** ([memoria EPS UAM 2021, listado](https://repositorio.uam.es/bitstreams/27cad248-b224-4b24-b508-431a2bf38f4e/download)) | TFM listado en UAM; no se localizo memoria completa en la busqueda | CXR | Deteccion COVID en radiografias | Muy util para demostrar que existen TFM cercanos en Espana | Segun el listado, parece centrado en CXR; no hay evidencia de CT, segmentacion, XAI comparativo ni matriz CXR/CT |
| IIC/UAM, **Deteccion COVID-19 en radiografias con Deep Learning** ([IIC](https://www.iic.uam.es/lasalud/deteccion-covid19-en-radiografias-con-deep-learning/)) | Proyecto tecnico/divulgativo, no TFM | CXR | Transfer learning, 3 clases, Grad-CAM | Antecedente nacional en castellano; util para contexto aplicado | No es memoria academica completa; no cubre CT ni segmentacion formal; advierte posible sesgo por calidad de imagen |
| Hosseini, 2023, **COVID-19 detection using chest X-ray images based on ML/DL** ([NMBU master thesis](https://nmbu.brage.unit.no/nmbu-xmlui/handle/11250/3079884)) | Master thesis | CXR | ML/DL y data augmentation | Buen antecedente de TFM internacional centrado en CXR | No integra CT, severidad, segmentacion ni XAI multimodal |
| ITU, 2021, **Deep feature transfer from deep learning models into machine learning algorithms...** ([Istanbul Technical University](https://research.itu.edu.tr/en/studentTheses/deep-feature-transfer-from-deep-learning-models-into-machine-lear)) | Master thesis | CXR | Transferencia de caracteristicas profundas + ML clasico | Sirve para mostrar aproximaciones hibridas | No aborda CT, segmentacion ni comparacion balanceo/XAI |
| Manandhar, 2023, **Identification of Coronavirus Through CT-scan Images Using Supervised and Semi-supervised Learning** ([NMBU master thesis](https://nmbu.brage.unit.no/nmbu-xmlui/handle/11250/3076768)) | Master thesis | CT | DenseNet201, ResNet50, CNNs, EfficientNetB4 semi-supervisado | Buen antecedente de CT | CT binario COVID/no-COVID; no CXR, no segmentacion/XAI y no severidad CT-0/CT-3+ |
| Ngong & Baykan, 2023, **CT classification + lesion segmentation cGAN-UNet** ([IIETA](https://www.iieta.org/journals/ts/paper/10.18280/ts.400101)) | Articulo derivado de trabajo academico; referencia a master thesis interna | CT | Segmentacion cGAN-UNet + clasificadores | Relacionado porque combina CT clasificacion y segmentacion | No incluye CXR ni comparativa entre modalidades; clasificacion binaria COVID/no-COVID |

Notas:

- El TFM de Garcia Astudillo aparece en una memoria institucional de la EPS-UAM, pero en esta busqueda no se encontro el documento completo. Conviene citarlo solo como antecedente tematico, no como comparacion metodologica detallada, salvo que se consiga la memoria completa.
- Las tesis NMBU e ITU si tienen ficha institucional y son utiles para la tabla de relacionados, aunque el peso principal del estado del arte deberia recaer en articulos revisados por pares.

---

## Matriz de huecos por dimension

| Dimension | Lo habitual en la literatura | Riesgo si tu TFM no lo discute | Como posicionar tu TFM |
|---|---|---|---|
| Modalidad | Muchos trabajos solo CXR o solo CT | Parecer un clasificador mas de COVID | Comparacion CXR vs CT bajo metodologia comun, aclarando que las etiquetas no son equivalentes clinicamente |
| Tarea | Clasificacion aislada | Estado del arte demasiado centrado en accuracy | Clasificacion + segmentacion + explicabilidad |
| Arquitecturas | Una arquitectura o varias sin misma matriz | Comparacion incompleta | Misma matriz ResNet-50, DenseNet-121, EfficientNet-B0 en CXR y CT |
| Desbalance | A veces se ignora o se compensa con submuestreo | Accuracy sesgada hacia clases dominantes | Baseline vs weighted CE vs focal loss; reportar F1-macro y matriz de confusion |
| CT | Frecuente binario COVID/no-COVID | Perder el valor de severidad | CT como severidad agrupada CT-0, CT-1, CT-2, CT-3+ y split por estudio |
| Segmentacion | U-Net/Attention U-Net en CT; CXR pulmon a veces como preprocesado | Mezclar pulmon con lesion | CXR: mascaras pulmonares; CT: lesion/infeccion. Interpretacion separada |
| XAI | Grad-CAM frecuente, pero a veces usado de forma superficial | Sobreinterpretar mapas de calor | Evaluar si la saliencia cae en regiones plausibles; aclarar que Grad-CAM/LIME/SHAP no son segmentaciones clinicas |
| Fuga de datos | Muchos trabajos mezclan fuentes o slices | Resultados inflados | En CT, split por `study_id`; en CXR, split estratificado y discusion de sesgo/shortcut learning |

---

## Tabla corta para incluir en la memoria

Esta version es mas compacta y puede adaptarse directamente al capitulo de estado del arte:

| Autor / trabajo | Modalidad | Tarea | Limitacion principal respecto a este TFM | Aportacion diferencial del TFM |
|---|---|---|---|---|
| Wang et al. - COVID-Net | CXR | Clasificacion | Arquitectura especifica y una sola modalidad | Comparacion controlada entre backbones estandar, CXR y CT |
| Constantinou et al. | CXR | Clasificacion TL | No incluye CT, segmentacion ni XAI | Matriz arquitectura x balanceo y ampliacion multimodal |
| Zebin & Rezvy | CXR | Clasificacion + Grad-CAM | XAI solo CXR, sin segmentacion formal | XAI comparado por modalidad y relacionado con mascaras |
| Rahman et al. / COVID Radiography DB | CXR | Dataset, enhancement, mascaras | Base CXR, no estudio CT comparativo | Uso del dataset dentro de un pipeline experimental mas amplio |
| MosMedData | CT | Dataset severidad | No es comparativa de modelos | Clasificacion CT reproducible con split por estudio |
| Qiblawey et al. | CT | Segmentacion + severidad | No incluye CXR ni matriz de balanceo | Integracion CXR/CT y lectura comparativa |
| Saood & Hatem | CT | Segmentacion | No clasificacion multimodal ni XAI | Segmentacion como fase dentro de un marco completo |
| Trivedi et al. | CXR | Shortcut learning | No compara CT ni segmentacion | Tu TFM incorpora discusion critica y evita basarse solo en accuracy |
| DeepCOVIDExplainer | CXR | XAI | No CT ni validacion con mascaras diferenciadas | Grad-CAM/LIME/SHAP interpretados con cautela anatomica |
| TFM/tesis CXR encontrados | CXR | Clasificacion | Alcance normalmente monomodal | Tu TFM combina CXR, CT, balanceo, segmentacion y XAI |

---

## Frase de cierre para defender la originalidad

Una formulacion prudente y fuerte:

> La originalidad del TFM no reside en proponer una nueva arquitectura aislada, sino en integrar en un mismo protocolo experimental varios elementos que suelen aparecer separados en la literatura: comparacion de backbones de transfer learning, manejo explicito del desbalance, dos modalidades de imagen con significado clinico distinto, segmentacion con mascaras disponibles y explicabilidad visual interpretada con cautela. Esta integracion permite obtener resultados no solo de rendimiento, sino tambien de robustez metodologica e interpretabilidad.

## Trabajos que conviene citar con prioridad

Prioridad alta para estado del arte:

1. Wang et al. COVID-Net.
2. Constantinou et al. clasificacion CXR con ResNet/DenseNet.
3. Rahman et al. COVID-19 Radiography Database y mascaras.
4. Morozov et al. MosMedData.
5. Qiblawey et al. severidad CT y segmentacion.
6. Zhou et al. Attention U-Net CT.
7. Trivedi et al. shortcut learning.
8. Selvaraju et al. Grad-CAM.
9. Karim et al. DeepCOVIDExplainer.
10. Lee et al. revision sistematica CXR/CT.

Prioridad media:

- Pham 2021 fine tuning CXR.
- Ucar/Korkmaz o Kc et al. DenseNet121 CXR.
- Saood & Hatem U-Net vs SegNet CT.
- Tesis NMBU/ITU como antecedentes de TFM.
