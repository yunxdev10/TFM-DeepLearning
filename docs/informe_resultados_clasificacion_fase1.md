# Informe de resultados de clasificacion - Fase 1

Fecha: 2026-05-14

Notebook analizado: `notebooks/04_classification_results.ipynb`

Artefactos usados:

- `results/classification/cxr/*_full_summary.json`
- `results/classification/cxr/*_full_classification_report.csv`
- `results/classification/cxr/*_full_confusion_matrix.csv`
- `results/classification/cxr/*_full_predictions.csv`
- `results/classification/ct/*_full_summary.json`
- `results/classification/ct/*_full_classification_report.csv`
- `results/classification/ct/*_full_confusion_matrix.csv`
- `results/classification/ct/*_full_predictions.csv`

## 1. Alcance del notebook 04

El notebook `04_classification_results.ipynb` consolida los resultados de clasificacion generados previamente por los notebooks de CXR y CT. Su objetivo no es entrenar modelos, sino comparar los artefactos ya guardados mediante:

- tabla resumen de metricas,
- seleccion del mejor modelo por dataset,
- grafico de barras de F1-macro por arquitectura y modalidad,
- matrices de confusion,
- curvas ROC one-vs-rest por clase,
- tabla cross-modal CXR vs CT.

Para la interpretacion cientifica se usan solamente los experimentos `full`. El notebook de resultados queda filtrado a artefactos completos para evitar mezclar pruebas tecnicas reducidas con evidencia experimental defendible.

## 2. Tabla comparativa principal

La tabla comparativa resume `accuracy`, `f1_macro`, `f1_weighted` y `auc_roc_macro` para cada combinacion de dataset, arquitectura y estrategia de balanceo.

### Resultados CXR full

| Experimento | Arquitectura | Balanceo | Accuracy | F1-macro | F1-weighted | AUC macro |
|---|---|---|---:|---:|---:|---:|
| `cxr_densenet121_weighted_ce` | DenseNet-121 | Weighted CE | 0.9496 | 0.9567 | 0.9496 | 0.9931 |
| `cxr_densenet121_focal_loss` | DenseNet-121 | Focal loss | 0.9455 | 0.9537 | 0.9455 | 0.9920 |
| `cxr_resnet50_weighted_ce` | ResNet-50 | Weighted CE | 0.9446 | 0.9512 | 0.9444 | 0.9923 |
| `cxr_densenet121_baseline` | DenseNet-121 | Baseline | 0.9458 | 0.9499 | 0.9458 | 0.9925 |
| `cxr_efficientnet_b0_weighted_ce` | EfficientNet-B0 | Weighted CE | 0.9402 | 0.9491 | 0.9400 | 0.9923 |
| `cxr_resnet50_baseline` | ResNet-50 | Baseline | 0.9436 | 0.9487 | 0.9433 | 0.9924 |
| `cxr_resnet50_focal_loss` | ResNet-50 | Focal loss | 0.9351 | 0.9434 | 0.9350 | 0.9898 |
| `cxr_efficientnet_b0_baseline` | EfficientNet-B0 | Baseline | 0.9398 | 0.9432 | 0.9397 | 0.9924 |
| `cxr_efficientnet_b0_focal_loss` | EfficientNet-B0 | Focal loss | 0.9348 | 0.9426 | 0.9346 | 0.9912 |

Interpretacion:

- CXR obtiene resultados altos y consistentes en todas las arquitecturas.
- Todas las combinaciones superan el umbral del 90% de accuracy previsto en el plan.
- El mejor resultado global es `cxr_densenet121_weighted_ce`, con 94.96% de accuracy y 95.67% de F1-macro.
- La diferencia entre los mejores modelos es pequena, lo que indica que el pipeline es estable para CXR.
- El AUC macro esta cerca de 0.99 en casi todos los modelos, lo que sugiere una separacion probabilistica muy fuerte entre clases.

### Resultados CT full

| Experimento | Arquitectura | Balanceo | Accuracy | F1-macro | F1-weighted | AUC macro |
|---|---|---|---:|---:|---:|---:|
| `ct_densenet121_baseline` | DenseNet-121 | Baseline | 0.6477 | 0.4173 | 0.6061 | 0.7229 |
| `ct_densenet121_weighted_ce` | DenseNet-121 | Weighted CE | 0.5449 | 0.4160 | 0.5546 | 0.6914 |
| `ct_resnet50_weighted_ce` | ResNet-50 | Weighted CE | 0.5215 | 0.4130 | 0.5373 | 0.7181 |
| `ct_resnet50_baseline` | ResNet-50 | Baseline | 0.6484 | 0.4043 | 0.6056 | 0.7200 |
| `ct_efficientnet_b0_baseline` | EfficientNet-B0 | Baseline | 0.6195 | 0.3897 | 0.5757 | 0.6943 |
| `ct_efficientnet_b0_weighted_ce` | EfficientNet-B0 | Weighted CE | 0.4862 | 0.3756 | 0.5063 | 0.6732 |
| `ct_efficientnet_b0_focal_loss` | EfficientNet-B0 | Focal loss | 0.4529 | 0.3731 | 0.4785 | 0.6903 |
| `ct_resnet50_focal_loss` | ResNet-50 | Focal loss | 0.4570 | 0.3693 | 0.4739 | 0.6993 |
| `ct_densenet121_focal_loss` | DenseNet-121 | Focal loss | 0.4407 | 0.3563 | 0.4548 | 0.6932 |

Interpretacion:

- CT presenta un rendimiento claramente inferior a CXR.
- El mejor resultado por accuracy es `ct_resnet50_baseline`, con 64.84%.
- El mejor resultado por F1-macro y AUC macro es `ct_densenet121_baseline`, con F1-macro 0.4173 y AUC macro 0.7229.
- La diferencia entre accuracy y F1-macro muestra que el modelo no funciona igual de bien en todas las clases.
- El objetivo inicial del plan para CT, accuracy igual o superior a 85%, no se alcanza. Esto debe presentarse como un hallazgo experimental: la clasificacion de severidad CT en slices 2D es considerablemente mas dificil que la clasificacion diagnostica CXR.

## 3. Mejor modelo por dataset

El notebook selecciona el mejor modelo por F1-macro, que es una decision adecuada porque hay desbalance de clases y porque F1-macro penaliza que el modelo ignore clases minoritarias.

| Dataset | Mejor modelo por F1-macro | Accuracy | F1-macro | F1-weighted | AUC macro |
|---|---|---:|---:|---:|---:|
| CXR | `cxr_densenet121_weighted_ce` | 0.9496 | 0.9567 | 0.9496 | 0.9931 |
| CT | `ct_densenet121_baseline` | 0.6477 | 0.4173 | 0.6061 | 0.7229 |

Interpretacion:

- DenseNet-121 es la arquitectura mas fuerte si se prioriza F1-macro.
- En CXR, DenseNet-121 con weighted CE mejora tanto la accuracy como el F1-macro.
- En CT, DenseNet-121 baseline no tiene la mayor accuracy por un margen minimo, pero si ofrece mejor equilibrio entre clases que ResNet-50 baseline.
- Para la memoria, conviene declarar dos lideres CT segun criterio:
  - mejor accuracy: `ct_resnet50_baseline`,
  - mejor equilibrio macro/AUC: `ct_densenet121_baseline`.

## 4. Grafico de barras: F1-macro por arquitectura y modalidad

El grafico de barras compara el F1-macro entre arquitecturas y modalidades. Su lectura principal es la brecha entre CXR y CT.

Interpretacion visual esperada:

- Las barras de CXR se situan alrededor de 0.94-0.96.
- Las barras de CT se situan alrededor de 0.36-0.42.
- La separacion entre modalidades es mucho mayor que la diferencia entre arquitecturas dentro de cada modalidad.

Promedios por arquitectura y dataset:

| Dataset | Arquitectura | Accuracy media | F1-macro medio | AUC macro medio |
|---|---|---:|---:|---:|
| CXR | DenseNet-121 | 0.9470 | 0.9534 | 0.9925 |
| CXR | ResNet-50 | 0.9411 | 0.9478 | 0.9915 |
| CXR | EfficientNet-B0 | 0.9383 | 0.9450 | 0.9920 |
| CT | DenseNet-121 | 0.5444 | 0.3965 | 0.7025 |
| CT | ResNet-50 | 0.5423 | 0.3955 | 0.7125 |
| CT | EfficientNet-B0 | 0.5195 | 0.3795 | 0.6859 |

Interpretacion:

- DenseNet-121 es la mejor arquitectura media en CXR.
- En CT, DenseNet-121 y ResNet-50 estan muy cerca en F1-macro medio.
- EfficientNet-B0 no aporta una ventaja clara en ninguno de los dos datasets.
- La conclusion principal del grafico no es que una arquitectura gane de forma absoluta, sino que la modalidad y el tipo de etiqueta condicionan mucho mas el rendimiento que la arquitectura.

## 5. Efecto de las estrategias de balanceo

Promedios por estrategia:

| Dataset | Estrategia | Accuracy media | F1-macro medio | AUC macro medio |
|---|---|---:|---:|---:|
| CXR | Weighted CE | 0.9448 | 0.9523 | 0.9926 |
| CXR | Baseline | 0.9431 | 0.9473 | 0.9924 |
| CXR | Focal loss | 0.9385 | 0.9466 | 0.9910 |
| CT | Baseline | 0.6385 | 0.4038 | 0.7124 |
| CT | Weighted CE | 0.5175 | 0.4015 | 0.6942 |
| CT | Focal loss | 0.4502 | 0.3662 | 0.6942 |

Interpretacion:

- En CXR, `weighted_ce` es la mejor estrategia media. Mejora F1-macro sin sacrificar AUC.
- En CT, `baseline` obtiene la mejor accuracy media y un F1-macro ligeramente superior a `weighted_ce`.
- `weighted_ce` en CT no mejora de forma clara el equilibrio entre clases, aunque en algunos modelos se acerca al baseline en F1-macro.
- `focal_loss` no mejora los resultados en esta fase; tiende a empeorar tanto CXR como CT.
- La focal loss puede estar penalizada por la configuracion concreta de hiperparametros, por la dificultad de CT o por la sensibilidad a clases minoritarias con pocas muestras.

## 6. Matrices de confusion CXR

La matriz de confusion mas importante para CXR es la del mejor modelo `cxr_densenet121_weighted_ce`.

Etiquetas:

- 0: COVID
- 1: Lung Opacity
- 2: Normal
- 3: Viral Pneumonia

Matriz de confusion:

| Real \\ Prediccion | COVID | Lung Opacity | Normal | Viral Pneumonia |
|---|---:|---:|---:|---:|
| COVID | 523 | 8 | 11 | 0 |
| Lung Opacity | 3 | 826 | 73 | 0 |
| Normal | 10 | 46 | 1470 | 3 |
| Viral Pneumonia | 0 | 0 | 6 | 196 |

Metricas por clase:

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| COVID | 0.9757 | 0.9649 | 0.9703 | 542 |
| Lung Opacity | 0.9386 | 0.9157 | 0.9270 | 902 |
| Normal | 0.9423 | 0.9614 | 0.9518 | 1529 |
| Viral Pneumonia | 0.9849 | 0.9703 | 0.9776 | 202 |

Interpretacion:

- El modelo clasifica muy bien las cuatro clases CXR.
- COVID y Viral Pneumonia tienen F1-score especialmente alto, por encima de 0.97.
- La mayor fuente de error aparece entre `Lung Opacity` y `Normal`:
  - 73 casos de Lung Opacity se predicen como Normal.
  - 46 casos de Normal se predicen como Lung Opacity.
- Este patron es clinicamente plausible, porque algunas opacidades pueden ser sutiles y compartir rasgos visuales con radiografias normales o de baja alteracion.
- Hay pocos errores entre COVID y Viral Pneumonia, lo que sugiere que el modelo separa razonablemente bien patrones de infeccion viral y patrones compatibles con COVID en este dataset.

Conclusion para CXR:

El rendimiento CXR es robusto, equilibrado y defendible como resultado principal positivo de la Fase 1. Las confusiones restantes se concentran en clases visualmente proximas y no invalidan la utilidad del modelo.

## 7. Matrices de confusion CT

Para CT hay dos modelos relevantes:

- `ct_resnet50_baseline`: mejor accuracy.
- `ct_densenet121_baseline`: mejor F1-macro y AUC macro.

Etiquetas:

- 0: CT-0
- 1: CT-1
- 2: CT-2
- 3: CT-3+

### CT - ResNet-50 baseline

Matriz de confusion:

| Real \\ Prediccion | CT-0 | CT-1 | CT-2 | CT-3+ |
|---|---:|---:|---:|---:|
| CT-0 | 427 | 584 | 1 | 0 |
| CT-1 | 261 | 2202 | 55 | 5 |
| CT-2 | 12 | 390 | 44 | 14 |
| CT-3+ | 11 | 118 | 10 | 21 |

Metricas por clase:

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CT-0 | 0.6006 | 0.4219 | 0.4956 | 1012 |
| CT-1 | 0.6685 | 0.8728 | 0.7571 | 2523 |
| CT-2 | 0.4000 | 0.0957 | 0.1544 | 460 |
| CT-3+ | 0.5250 | 0.1313 | 0.2100 | 160 |

Interpretacion:

- El modelo se concentra fuertemente en `CT-1`, la clase mayoritaria.
- `CT-1` alcanza recall 0.8728, pero `CT-2` y `CT-3+` tienen recall muy bajo.
- Muchos casos `CT-0`, `CT-2` y `CT-3+` son absorbidos por `CT-1`.
- Los errores mas importantes son:
  - CT-0 -> CT-1: 584 casos.
  - CT-2 -> CT-1: 390 casos.
  - CT-1 -> CT-0: 261 casos.
  - CT-3+ -> CT-1: 118 casos.
- La accuracy de 0.6484 se explica en gran parte por el buen rendimiento en `CT-1`, no por una clasificacion equilibrada de severidad.

### CT - DenseNet-121 baseline

Matriz de confusion:

| Real \\ Prediccion | CT-0 | CT-1 | CT-2 | CT-3+ |
|---|---:|---:|---:|---:|
| CT-0 | 446 | 566 | 0 | 0 |
| CT-1 | 300 | 2178 | 33 | 12 |
| CT-2 | 22 | 384 | 36 | 18 |
| CT-3+ | 6 | 113 | 10 | 31 |

Metricas por clase:

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CT-0 | 0.5762 | 0.4407 | 0.4994 | 1012 |
| CT-1 | 0.6720 | 0.8633 | 0.7557 | 2523 |
| CT-2 | 0.4557 | 0.0783 | 0.1336 | 460 |
| CT-3+ | 0.5082 | 0.1938 | 0.2805 | 160 |

Interpretacion:

- DenseNet-121 baseline mejora ligeramente el equilibrio macro respecto a ResNet-50.
- Detecta algo mejor `CT-3+` que ResNet-50, con recall 0.1938 frente a 0.1313.
- Sigue fallando de forma severa en `CT-2`, con recall 0.0783.
- La clase `CT-1` domina el comportamiento del modelo.
- El F1-macro bajo indica que el modelo no resuelve adecuadamente la ordenacion de severidad CT.

Conclusion para CT:

La matriz de confusion muestra que CT no falla de forma aleatoria: el error tiene una direccion clara hacia `CT-1`. Esto sugiere que el modelo aprende un patron central o mayoritario, pero no separa bien severidad intermedia y severa. Para la memoria, este resultado debe discutirse como limitacion metodologica del enfoque 2D por slice y del desbalance de severidad, no como simple falta de capacidad de la red.

## 8. Curvas ROC multiclase

El notebook genera curvas ROC one-vs-rest para cada clase y cada experimento. Estas curvas miden la capacidad de separar cada clase frente al resto usando las probabilidades del modelo.

### Mejor CXR: DenseNet-121 weighted CE

| Clase | AUC OvR |
|---|---:|
| COVID | 0.9988 |
| Lung Opacity | 0.9866 |
| Normal | 0.9870 |
| Viral Pneumonia | 0.9998 |

Interpretacion:

- Todas las clases CXR tienen AUC muy alto.
- COVID y Viral Pneumonia son casi perfectamente separables en terminos probabilisticos.
- Lung Opacity y Normal son las clases con AUC algo menor, coherente con la matriz de confusion.
- La ROC confirma que el modelo no solo acierta la etiqueta final, sino que asigna probabilidades utiles y separables.

### Mejor CT por accuracy: ResNet-50 baseline

| Clase | AUC OvR |
|---|---:|
| CT-0 | 0.8087 |
| CT-1 | 0.6870 |
| CT-2 | 0.7328 |
| CT-3+ | 0.6513 |

Interpretacion:

- CT tiene AUC moderado, no inutil, pero claramente inferior a CXR.
- CT-0 es la clase mas separable.
- CT-3+ es la mas dificil en este modelo.
- El AUC de CT-2 es razonable comparado con su recall bajo, lo que sugiere que el modelo tiene cierta informacion probabilistica, pero el umbral/decision final no convierte esa informacion en predicciones correctas frecuentes.

### Mejor CT por F1-macro: DenseNet-121 baseline

| Clase | AUC OvR |
|---|---:|
| CT-0 | 0.8070 |
| CT-1 | 0.6776 |
| CT-2 | 0.7367 |
| CT-3+ | 0.6703 |

Interpretacion:

- DenseNet-121 baseline ofrece el mayor AUC macro total en CT.
- Mejora algo la separacion de CT-3+ frente a ResNet-50.
- Aun asi, la separacion de CT-1 y CT-3+ esta lejos de ser fuerte.
- Las ROC apoyan la misma conclusion que las matrices: CT contiene senal, pero no suficiente para una clasificacion de severidad robusta en este planteamiento.

## 9. Lectura cross-modal CXR vs CT

La tabla cross-modal compara CXR y CT para las mismas arquitecturas y estrategias. Debe interpretarse con cuidado porque las tareas no son clinicamente equivalentes:

- CXR clasifica diagnostico radiografico en cuatro categorias.
- CT clasifica severidad agrupada, con CT-3 y CT-4 fusionadas en CT-3+.

Interpretacion:

- CXR supera claramente a CT en todas las arquitecturas y estrategias.
- La diferencia no debe leerse como "CXR es siempre mejor que CT" en medicina, sino como que esta tarea concreta de clasificacion CXR es mucho mas favorable para transfer learning que la tarea CT planteada.
- CT tiene mayor dificultad por varios factores:
  - etiquetas de severidad mas graduales,
  - fuerte predominio de CT-1,
  - clases graves con menos ejemplos,
  - uso de slices 2D que pueden no representar toda la informacion volumetrica del estudio,
  - posible ruido entre etiqueta por estudio y contenido visible de cada slice.

## 10. Respuesta a las preguntas de investigacion de Fase 1

### RQ1: Que arquitectura funciona mejor?

Para CXR, DenseNet-121 es la mejor arquitectura. Su mejor configuracion es `weighted_ce`, con F1-macro 0.9567.

Para CT, DenseNet-121 es ligeramente mejor si se prioriza F1-macro y AUC macro, mientras que ResNet-50 tiene una accuracy apenas superior. La diferencia es pequena, por lo que no se puede afirmar una superioridad fuerte de una arquitectura en CT.

### RQ2: Que estrategia de balanceo funciona mejor?

En CXR, `weighted_ce` es la estrategia mas efectiva. Mejora el F1-macro medio y produce el mejor modelo global.

En CT, el baseline funciona igual o mejor que las estrategias de balanceo. `weighted_ce` no corrige de forma suficiente los errores en clases minoritarias y `focal_loss` empeora los resultados en esta configuracion.

## 11. Conclusiones principales

1. La Fase 1 queda validada para CXR: los modelos alcanzan rendimiento alto, estable y competitivo.

2. El mejor modelo CXR es `cxr_densenet121_weighted_ce`, con accuracy 0.9496, F1-macro 0.9567 y AUC macro 0.9931.

3. CT es una tarea mucho mas dificil: el mejor modelo no supera el 65% de accuracy y el F1-macro queda alrededor de 0.42.

4. En CT, la clase `CT-1` domina el aprendizaje y absorbe muchos ejemplos de `CT-0`, `CT-2` y `CT-3+`.

5. La diferencia CXR vs CT debe interpretarse metodologicamente: no compara dos modalidades clinicas equivalentes, sino dos tareas con distinta naturaleza de etiqueta, distinto desbalance y distinto nivel de dificultad.

6. La focal loss no aporta mejora en esta fase. Weighted CE ayuda en CXR, pero no resuelve CT.

7. Para los capitulos posteriores, los modelos finales recomendados son:
   - CXR: `cxr_densenet121_weighted_ce_full`.
   - CT: `ct_densenet121_baseline_full` si se prioriza F1-macro/AUC; `ct_resnet50_baseline_full` si se reporta por accuracy.

## 12. Texto redactable para la memoria

Los experimentos de clasificacion muestran una diferencia clara entre las dos modalidades analizadas. En el dataset CXR, los modelos preentrenados alcanzan un rendimiento alto y estable, con DenseNet-121 y weighted cross-entropy como mejor configuracion. Este modelo obtiene una accuracy de 0.9496, un F1-macro de 0.9567 y un AUC macro de 0.9931, superando el objetivo experimental previsto para la fase. Las matrices de confusion indican que los errores principales se concentran entre las clases Normal y Lung Opacity, un patron coherente con la proximidad visual de ambas categorias.

En cambio, la clasificacion CT de severidad presenta un rendimiento considerablemente menor. El mejor modelo por accuracy es ResNet-50 baseline, con 0.6484, mientras que DenseNet-121 baseline obtiene el mejor F1-macro y AUC macro. Las matrices de confusion muestran que los modelos tienden a predecir la clase CT-1, lo que produce recalls bajos en CT-2 y CT-3+. Este comportamiento evidencia la dificultad de la tarea CT, probablemente asociada al desbalance de clases, la naturaleza gradual de las etiquetas de severidad, el uso de slices 2D y la menor representacion de casos graves. Por tanto, los resultados CT deben interpretarse como una limitacion relevante del enfoque y como motivacion para las fases posteriores de segmentacion y explicabilidad.

## 13. Proximos pasos

- Guardar este informe como cierre interpretativo de Fase 1.
- Ejecutar o refrescar visualmente el notebook `04_classification_results.ipynb` si se quieren conservar las salidas dentro del `.ipynb`.
- Empezar Fase 2 con `src/models/segmentation.py` y `notebooks/05_segmentation.ipynb`.
- Usar `cxr_densenet121_weighted_ce_full` y `ct_densenet121_baseline_full` como modelos candidatos para XAI, salvo que se decida priorizar CT por accuracy y usar `ct_resnet50_baseline_full`.
