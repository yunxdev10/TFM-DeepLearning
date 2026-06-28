# Revision de `main (2).pdf`

Fecha: 2026-05-30

PDF revisado: `/Users/yuncaichen/Downloads/main (2).pdf`

## Veredicto general

La version revisada esta bien orientada en la introduccion, objetivos y estado del arte. En lo conceptual, el documento ya refleja bastante bien lo que se hizo en el TFM: dos modalidades, CXR y CT; clasificacion con ResNet-50, DenseNet-121 y EfficientNet-B0; segmentacion con U-Net y Attention U-Net; mejora experimental de CT con Tversky, parches positivos, contexto mixto, 2.5D, ensemble y ajuste de capacidad; y explicabilidad con Grad-CAM.

Sin embargo, todavia no es una version completa defendible. A partir de la pagina 32, la metodologia queda casi vacia y las secciones de resultados, discusion, conclusiones y trabajo futuro aparecen como titulos sin desarrollo. El indice muestra muchas secciones con la misma pagina 34, lo que confirma que esas partes estan pendientes.

El punto mas importante: no hay incoherencias graves de "decir que hicimos algo que no hicimos" en las partes revisadas. No se afirma que se haya usado ResUNet, SHAP o LIME. Tampoco se presenta 3D U-Net como metodologia propia; aparece como contexto del estado del arte, lo cual es correcto si se mantiene asi.

## Correcciones de prioridad alta

### 1. Sustituir portada provisional

En la primera pagina aparecen:

- `Titulo provisional del TFM`
- `Nombre del autor`

Esto debe sustituirse por el titulo definitivo, nombre, tutor/es, titulacion, universidad y fecha formal. No puede quedar como marcador provisional.

### 2. Completar metodologia

Ahora mismo solo esta desarrollada la subseccion `3.1.1. Aumento de datos`. Las secciones siguientes aparecen solo como encabezados:

- `3.2. Arquitecturas de clasificacion`
- `3.3. Arquitecturas de segmentacion`
- `3.4. Estrategia de transfer learning y fine-tuning`
- `3.5. Manejo del desbalanceo`
- `3.6. Metodos de explicabilidad`
- `3.7. Metricas de evaluacion`
- `3.8. Diseno experimental`

Esta es la parte que mas urge completar. Debe explicar exactamente que se hizo, no solo la teoria. Debe incluir:

- Datasets reales usados: COVID-19 Radiography Database para CXR y MosMedData para CT.
- Diferencia entre etiquetas CXR y CT.
- Particiones train/validation/test y control de fuga, especialmente separacion por estudio en CT.
- Preprocesamiento de imagenes, canales, redimensionado y normalizacion.
- Clasificacion: ResNet-50, DenseNet-121, EfficientNet-B0 con transfer learning.
- Estrategias de balanceo: baseline, weighted cross-entropy, focal loss y oversampling; augmentacion como regularizacion/apoyo, no como fuente de casos clinicos reales.
- Segmentacion: U-Net y Attention U-Net.
- CXR: segmentacion pulmonar.
- CT: segmentacion de lesion/infeccion.
- Variantes CT: Tversky+BCE, `pos_weight`, positive crop sampling, patch-based training, mixed context, 2.5D, ensemble, threshold en validacion y `base_features`.
- Grad-CAM como unico metodo XAI implementado.
- Calibracion si se va a incluir el notebook 08: ECE, MCE, Brier, NLL, diagramas de fiabilidad y errores de alta confianza.

### 3. Completar resultados con cifras reales

La seccion `4. Resultados` esta vacia. Debe incluir las tablas y figuras finales ya generadas en `results/final_analysis/` y `results/calibration/`.

Resultados clave que deben aparecer:

| Bloque | Resultado principal |
|---|---|
| CXR clasificacion | Mejor modelo: `cxr_densenet121_weighted_ce`; accuracy `0.9496`, F1-macro `0.9567`, AUC macro `0.9931`. |
| CT clasificacion | Mejor accuracy: `ct_resnet50_baseline`, accuracy `0.6484`. Mejor F1/AUC: `ct_densenet121_baseline`, F1-macro `0.4173`, AUC macro `0.7229`. |
| CXR segmentacion | Mejor modelo: `cxr_attention_unet_segmentation`; Dice `0.9853`, IoU `0.9715`. |
| CT segmentacion | Mejor modelo individual final: `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`; Dice `0.5637`, IoU `0.4305`. |
| XAI CXR | Grad-CAM vs mascara pulmonar: IoU medio `0.2255`, ratio dentro de mascara `0.3143`. |
| XAI CT | Grad-CAM vs mascara de infeccion: mejor IoU observado `0.0146`, pico dentro de mascara `0.0000`. |
| Calibracion CXR | `cxr_densenet121_weighted_ce`: ECE `0.0109`, Brier `0.0801`. |
| Calibracion CT | `ct_resnet50_baseline`: ECE `0.0409`, Brier `0.4870`, 42 errores con confianza `>=0.90`; `ct_densenet121_baseline`: ECE `0.0214`, Brier `0.4844`, 14 errores de alta confianza. |

### 4. Cambiar el titulo de `4.6. Segmentacion de lesiones`

Ese titulo es incompleto porque en CXR no segmentamos lesiones, sino pulmones. Conviene cambiarlo por:

`4.6. Segmentacion pulmonar en CXR y segmentacion de lesion en CT`

Esto evita una incoherencia importante.

### 5. Anadir calibracion si el notebook 08 forma parte del TFM

El estado del arte ya menciona calibracion de forma breve, y el proyecto tiene `notebooks/08_calibration_analysis.ipynb`. Si se quiere incluir como aportacion final, debe aparecer en:

- Metodologia: una subseccion de calibracion probabilistica.
- Resultados: metricas ECE, MCE, Brier, NLL y errores de alta confianza.
- Discusion: diferencia entre rendimiento y confianza, especialmente en CT.
- Conclusiones: CXR no solo rinde mejor, tambien esta mejor calibrado; CT presenta mas incertidumbre y errores de alta confianza.

Si no se quiere incluir como fase principal, debe presentarse como analisis complementario, no como objetivo central.

### 6. Corregir errores de redaccion y espacios

El PDF muestra muchos problemas de espaciado, probablemente por LaTeX o por comandos `\textit{}` pegados a palabras. Ejemplos:

- `ElobjetivogeneraldeesteTFM`
- `dedeep learningcontransfer learning`
- `modelos dedeep learningcontransfer learning`
- `Eloversamplingmodificalafrecuencia...`
- `Ensegmentacionbiomedica...`
- `base features` aparece pegado como `ybase features`

Hay que revisar el fuente y anadir espacios antes/despues de comandos de formato, por ejemplo:

- `de \textit{deep learning} con \textit{transfer learning}`
- `El \textit{oversampling} modifica...`
- `En segmentacion biomedica...`
- `capacidad del modelo y \textit{base features}`

## Correcciones por seccion

### Portada e indice

Estado: incompleto.

Correcciones:

- Sustituir titulo y autor provisionales.
- Revisar que el indice no lleve Metodologia, Resultados, Discusion y Conclusiones todos a la pagina 34 cuando el documento este completo.
- Revisar nombres de secciones para que coincidan con lo implementado.

### Introduccion

Estado: conceptualmente correcta, pero necesita limpieza linguistica.

Cumple con el TFM porque:

- Presenta CXR y CT como modalidades distintas y no equivalentes.
- Justifica clasificacion, segmentacion, desbalanceo y Grad-CAM.
- Menciona correctamente los dos datasets: COVID-19 Radiography Database y MosMedData.
- No promete SHAP/LIME.

Correcciones concretas:

- `La CXR al ser una tecnica rapida...` debe reescribirse.
- `amplicamente`, `escanarios`, `clinica`, `analisis`, `automatico`, `imagenes medicas`, `segementacion` tienen errores ortograficos o faltan acentos.
- `radion` debe ser `radiacion`.
- `la informacion clinica que proporcionan no es del todo coherente` no es preciso. Mejor: `la informacion clinica que proporcionan no es directamente equivalente`.
- `Esto se aplican` debe ser `Esto se aplica`.

Texto recomendado para sustituir el parrafo CXR/CT:

> Las dos modalidades de imagen consideradas en este trabajo, la radiografia de torax (CXR) y la tomografia computarizada (CT), ofrecen informacion complementaria. La CXR es una tecnica rapida, accesible y de bajo coste, ampliamente utilizada en escenarios de cribado, seguimiento y priorizacion clinica. Sin embargo, su resolucion anatomica es limitada y puede dificultar la visualizacion de lesiones sutiles. La CT, por el contrario, proporciona mayor resolucion anatomica y permite observar con mas detalle patrones como opacidades en vidrio deslustrado, consolidaciones o afectacion bilateral. Esta ventaja implica mayor dosis de radiacion, mayor coste y una interpretacion mas dependiente del contexto clinico.

Texto recomendado para la comparacion entre modalidades:

> Por tanto, la comparacion entre CXR y CT debe interpretarse con cautela. Las etiquetas diagnosticas de CXR y los grados de severidad CT no representan exactamente la misma informacion clinica. En este TFM se comparan dentro de un marco metodologico comun, pero no se asume equivalencia clinica directa entre ambas modalidades.

### Objetivos y preguntas de investigacion

Estado: bien alineado.

Cumple con lo hecho porque:

- Clasificacion con ResNet-50, DenseNet-121 y EfficientNet-B0.
- Segmentacion con U-Net y Attention U-Net.
- Grad-CAM como metodo XAI.
- Comparacion saliencia-mascara diferenciando CXR y CT.
- Variantes CT reflejan los experimentos reales.

Ajustes recomendados:

- En el objetivo sobre balanceo, matizar `aumento de datos` como apoyo/regularizacion, no como tecnica principal aislada si no se presenta un experimento independiente de augmentacion.
- Si se incluye calibracion como parte final, anadir un objetivo complementario:

> Evaluar la calibracion probabilistica de los clasificadores mediante ECE, MCE, Brier score, NLL y analisis de errores de alta confianza.

- Si se quiere una RQ nueva:

> RQ6: ¿Hasta que punto la confianza probabilistica de los clasificadores refleja su rendimiento real en CXR y CT?

Tambien puede incluirse calibracion sin crear RQ nueva, como analisis complementario.

### Estado del arte

Estado: bastante solido y fiel al proyecto.

Puntos correctos:

- La explicacion de CXR, CT, slices 2D y 2.5D es relevante.
- La distincion entre etiqueta y mascara esta bien planteada.
- U-Net, Attention U-Net, encoder-decoder, skip connections, Dice, IoU, BCE y Tversky estan justificados.
- Grad-CAM aparece como tecnica principal, no SHAP/LIME.
- La parte de desbalanceo cubre baseline, weighted CE, focal loss, oversampling, augmentacion y desbalance pixel a pixel.
- `nnU-Net`, `3D U-Net` y modelos 3D aparecen como contexto bibliografico, no como desarrollo propio. Esto es correcto.

Correcciones de redaccion:

- Cambiar `esta tesis de master` por `este TFM` para mantener estilo.
- Corregir `muchas trabajos convertien los volumenes` por `muchos trabajos convierten los volumenes`.
- Corregir `Entendemos como etiqueta de clase como...` por `Una etiqueta de clase es...`.
- Corregir `mascara` por `mascara` con acento si el documento usa acentos: `mascara`.
- Corregir frases pegadas por comandos LaTeX: `pooling`, `flatten`, `softmax`, `sigmoid`, `backbone`, `fine-tuning`, `early stopping`, `skip connections`.

### Metodologia

Estado: insuficiente.

Debe escribirse como una receta reproducible de lo ejecutado. Estructura recomendada:

1. `3.1. Datasets`
   - CXR: COVID-19 Radiography Database, cuatro clases.
   - CT: MosMedData, grados CT-0 a CT-4, fusion CT-3/CT-4 como CT-3+ si se mantiene.
   - Mascaras: pulmonares en CXR; infeccion/lesion en CT.

2. `3.2. Preprocesamiento y particiones`
   - Redimensionado, normalizacion, conversion de canales.
   - Split train/validation/test.
   - Separacion por estudio en CT.
   - Uso exclusivo de train para augmentacion.

3. `3.3. Clasificacion`
   - ResNet-50, DenseNet-121, EfficientNet-B0.
   - Transfer learning, cambio de cabeza clasificadora.
   - Hiperparametros principales y early stopping.

4. `3.4. Desbalanceo`
   - Baseline.
   - Weighted cross-entropy.
   - Focal loss.
   - Oversampling/WeightedRandomSampler.
   - Matizar augmentacion.

5. `3.5. Segmentacion`
   - U-Net y Attention U-Net.
   - CXR pulmon.
   - CT lesion/infeccion.
   - Losses: Dice+BCE, weighted Tversky+BCE.
   - Parches positivos, contexto mixto, threshold en validacion, 2.5D, ensemble, base features.

6. `3.6. Grad-CAM`
   - Modelos explicados.
   - Capa objetivo.
   - Normalizacion/binarizacion de saliencia.
   - Comparacion con mascaras.

7. `3.7. Calibracion`
   - Solo si se incluye notebook 08.
   - ECE, MCE, Brier, NLL, reliability diagrams, errores alta confianza.

8. `3.8. Diseno experimental`
   - Resultados `full` como base de la memoria.
   - Seleccion en validacion y evaluacion final en test.
   - Trazabilidad de artefactos.

### Resultados

Estado: vacio.

Debe construirse desde los artefactos finales:

- `results/final_analysis/classification_results_with_ci.csv`
- `results/final_analysis/classification_best_by_accuracy.csv`
- `results/final_analysis/classification_best_by_f1.csv`
- `results/final_analysis/classification_mcnemar_top2.csv`
- `results/final_analysis/segmentation_results.csv`
- `results/final_analysis/segmentation_best_by_dice.csv`
- `results/final_analysis/xai_gradcam_results.csv`
- `results/calibration/calibration_metrics.csv`

Orden recomendado:

1. Analisis exploratorio.
2. Clasificacion CXR.
3. Clasificacion CT.
4. Comparacion CXR vs CT.
5. Estrategias de balanceo.
6. Segmentacion CXR/CT.
7. Mejoras CT y ablaciones.
8. Grad-CAM y comparacion con mascaras.
9. Calibracion.

### Discusion

Estado: vacia.

Debe responder a las RQ con una lectura critica:

- RQ1: DenseNet-121 con weighted CE domina CXR; CT no tiene ganador unico, ResNet-50 gana en accuracy y DenseNet-121 en F1/AUC.
- RQ2: CXR es mucho mas estable que CT; la CT de severidad por slices 2D es mas dificil.
- RQ3: Grad-CAM en CXR muestra plausibilidad anatomica parcial; en CT no se alinea con lesion.
- RQ4: Attention U-Net es la familia mas fuerte; CXR alcanza rendimiento muy alto, CT mejora pero sigue limitado.
- RQ5: weighted CE ayuda en CXR; en CT las estrategias de balanceo no superan claramente el baseline.
- Si se incluye RQ6: CXR tiene mejor calibracion; CT presenta mas errores de alta confianza y mayor incertidumbre.

### Conclusiones y trabajo futuro

Estado: vacio.

Conclusiones defendibles:

- El pipeline CXR funciona muy bien para clasificacion y segmentacion pulmonar.
- CT es mas dificil por severidad, variabilidad entre slices, desbalance y limitaciones de contexto 2D.
- Las mejoras CT son reales y experimentales: Tversky, mixed context, ensemble y mayor capacidad mejoran frente al baseline.
- Grad-CAM es util como auditoria visual, pero no valida causalidad ni sustituye segmentacion.
- La comparacion saliencia-mascara es una aportacion metodologica prudente.
- La calibracion aporta una capa adicional para interpretar confianza.

Trabajo futuro:

- Validacion externa multicentrica.
- Modelos 3D o 2.5D mas avanzados para CT.
- Cross-validation por estudio.
- Calibracion con temperature scaling.
- Comparacion con metodos XAI adicionales solo si se implementan realmente.
- Evaluacion clinica experta de mapas Grad-CAM y mascaras.

## Coherencia con lo realmente realizado

| Tema | Estado en PDF | Correccion necesaria |
|---|---|---|
| Dos datasets CXR/CT | Correcto | Mantener. |
| CXR segmentacion pulmonar | Correcto en texto conceptual | Cambiar titulo `Segmentacion de lesiones` en resultados. |
| CT segmentacion lesion/infeccion | Correcto | Mantener. |
| U-Net y Attention U-Net | Correcto | Mantener. |
| ResUNet | No aparece como usado | Correcto, no anadir. |
| 3D U-Net | Aparece como contexto | Correcto si no se presenta como experimento propio. |
| 2.5D CT | Correcto | En resultados explicar que no fue el mejor final. |
| Grad-CAM | Correcto | Mantener como unico XAI implementado. |
| SHAP/LIME | No aparecen como implementados | Correcto, no anadir como metodologia/resultados. |
| Desbalanceo | Correcto | Matizar augmentacion. |
| Calibracion | Aparece mencionada parcialmente | Decidir si se incluye como analisis complementario o RQ nueva. |
| Resultados | No desarrollados | Completar con tablas/figuras reales. |
| Discusion/conclusion | No desarrolladas | Completar con respuestas a RQ. |

## Checklist antes de entregar

- [ ] Portada definitiva.
- [ ] Resumen/abstract si la plantilla lo exige.
- [ ] Introduccion corregida linguisticamente.
- [ ] Objetivos ajustados si se incluye calibracion.
- [ ] Estado del arte revisado para espacios, acentos y terminos ingleses.
- [ ] Metodologia completa y reproducible.
- [ ] Resultados completos con cifras reales.
- [ ] Figuras y tablas insertadas con pie explicativo.
- [ ] Discusion respondiendo RQ1-RQ5 y, si aplica, RQ6.
- [ ] Limitaciones explicitas: dataset, validacion externa, slices 2D, Grad-CAM no causal.
- [ ] Conclusiones y trabajo futuro.
- [ ] Bibliografia revisada: todos los `cite` deben aparecer y todas las referencias usadas deben estar citadas.
- [ ] Revisar que no se diga que se implemento SHAP, LIME, ResUNet o 3D U-Net.
- [ ] Revisar que la memoria reporte solo resultados `full`.

## Prioridad de trabajo recomendada

1. Corregir redaccion y espacios en introduccion/estado del arte.
2. Escribir metodologia real.
3. Volcar resultados finales desde `results/final_analysis/` y `results/calibration/`.
4. Escribir discusion por preguntas de investigacion.
5. Cerrar conclusiones y trabajo futuro.
