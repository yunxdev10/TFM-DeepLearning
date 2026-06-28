# Informe de mejora adicional para segmentacion CT

Fecha: 2026-05-15

## 1. Situacion actual

El mejor modelo CT entrenado hasta ahora es:

| Experimento | Dice | IoU | Pixel accuracy | Threshold guardado |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.8 |
| `ct_attention_unet_segmentation_full` | 0.5001 | 0.3648 | 0.9960 | 0.5 |
| `ct_unet_segmentation_full` | 0.4790 | 0.3522 | 0.9971 | 0.5 |

La mejora mixed-context frente al baseline Attention U-Net es real: +0.0241 Dice y +0.0198 IoU.

## 2. Lectura del problema

La segmentacion CT sigue siendo dificil por tres razones:

- La lesion ocupa muy pocos pixeles. En test, el promedio es aproximadamente 229 pixeles por slice sobre 65.536 pixeles.
- El rendimiento cambia mucho segun el estudio. En el split actual, `study_0285`, `study_0268` y `study_0294` son claramente mas dificiles que `study_0274` y `study_0303`.
- Las lesiones pequenas dominan el error: cuanto menor es el area real de lesion, mas bajo cae Dice.

Rendimiento por tamano de lesion en test:

| Area real de lesion | N slices | Dice baseline Attention | Dice mixed-context, threshold 0.90 diagnostico |
|---|---:|---:|---:|
| <=50 px | 18 | 0.3072 | 0.3720 |
| 51-150 px | 36 | 0.4522 | 0.5063 |
| 151-300 px | 26 | 0.5170 | 0.5426 |
| 301-600 px | 22 | 0.6423 | 0.6567 |
| 601-1200 px | 8 | 0.7036 | 0.7223 |

Interpretacion:

- Mixed-context ayuda sobre todo en lesiones pequenas y medianas.
- Aun asi, el peor grupo sigue siendo `<=50 px`; ahi esta el principal margen de mejora.
- El modelo mixed-context con threshold 0.90 predice de media 285 pixeles, mas cerca de los 229 reales que el baseline con threshold 0.50, que predice 373 pixeles. Esto sugiere menos sobresegmentacion.

## 3. Umbral y postprocesado

El barrido diagnostico muestra:

| Modelo | Mejor threshold test diagnostico | Dice test diagnostico | IoU test diagnostico |
|---|---:|---:|---:|
| Baseline Attention U-Net | 0.45-0.50 | 0.5003 | 0.3648 |
| Mixed-context | 0.90 | 0.5387 | 0.4003 |
| Mixed-context + cierre morfologico ligero | 0.90 | 0.5389 | 0.4007 |

Esta lectura debe usarse con cuidado:

- El threshold `0.90` en test es diagnostico, no debe venderse como resultado final principal si no se selecciona en validacion o cross-validation.
- El postprocesado aporta una mejora minima sobre mixed-context: +0.0002 Dice aproximadamente. No merece priorizarse.
- Test-time augmentation horizontal no mejora el conjunto completo: baja de 0.5387 a 0.5297 en el analisis diagnostico.

## 4. Cambio recomendado

La mejora mas defendible ahora es probar CT 2.5D:

- Entrada de 3 canales:
  - canal 1: slice anterior `z-1`,
  - canal 2: slice actual `z`,
  - canal 3: slice siguiente `z+1`.
- Misma mascara objetivo del slice actual.
- Mismo split por estudio.
- Validacion y test siguen usando slices completos, no patches recortados.

Justificacion:

- CT es volumetrico, pero el modelo actual es 2D puro.
- Las lesiones COVID suelen extenderse entre slices cercanos.
- El analisis muestra que el fallo principal esta en lesiones pequenas o ambiguas, donde el contexto de slices vecinos puede ayudar.

Se creo el notebook:

- `notebooks/05f_ct_25d_context_training.ipynb`

Configuracion inicial:

- `Attention U-Net`
- `in_channels = 3`
- `base_features = 16`
- `weighted_tversky_bce`
- `pos_weight = 20`
- `train_crop_size = (160, 160)`
- `train_crop_prob = 0.5`
- `positive_crop_prob = 0.8`
- `epochs = 18`
- `optimize_threshold = True`

## 5. Plan de experimentacion

Orden recomendado:

1. Ejecutar `05f_ct_25d_context_training.ipynb`.
2. Comparar contra mixed-context 2D, no contra U-Net simple.
3. Si mejora Dice/IoU:
   - probar `ct25d_mixed30_patch192_pos70_tversky_pos10_thr`,
   - probar `ct25d_full_context_tversky_pos10_thr`.
4. Si no mejora:
   - reportar 2.5D como experimento negativo,
   - mantener mixed-context 2D como mejor modelo CT,
   - reforzar la defensa con analisis por tamano de lesion y por estudio.

No se recomienda priorizar:

- pixel accuracy, porque esta dominada por fondo;
- mas postprocesado morfologico, porque la ganancia es marginal;
- TTA horizontal, porque no mejora el promedio;
- patch-only training, porque ya empeoro el baseline.

## 6. Conclusiones defendibles

La linea experimental CT queda bien estructurada:

1. Baseline U-Net y Attention U-Net.
2. Loss ponderada Tversky/BCE.
3. Patch training positivo.
4. Mixed-context training.
5. Analisis de umbral/postprocesado.
6. Propuesta 2.5D con contexto volumetrico local.

Aunque CT no llegue a resultados tipo CXR, el TFM tiene una narrativa experimental solida: se identifica el desbalance, se comprueban alternativas, se descartan las que no ayudan y se propone una mejora alineada con la naturaleza volumetrica del dato.

## 7. Resultado de CT 2.5D

Tras ejecutar `notebooks/05f_ct_25d_context_training.ipynb`, el resultado fue:

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| Mixed-context 2D | 0.5242 | 0.3846 | 0.9959 | 0.80 |
| CT 2.5D mixed-context | 0.5143 | 0.3843 | 0.9972 | 0.95 |

Interpretacion:

- CT 2.5D no supera al mejor mixed-context 2D.
- IoU queda practicamente empatado, pero Dice baja de `0.5242` a `0.5143`.
- El threshold `0.95` y el aumento de pixel accuracy indican una prediccion mas conservadora: el modelo predice menos pixeles positivos.
- En test, CT 2.5D predice de media `249.7` pixeles frente a `399.8` del mixed-context 2D, con un objetivo real medio de `229.3`.
- Esta mayor conservacion ayuda en algunos estudios, como `study_0263` y `study_0294`, pero perjudica otros, especialmente `study_0268`, `study_0274` y el caso aislado `study_0287`.

Comparacion por area de lesion:

| Area real de lesion | Dice mixed-context 2D | Dice CT 2.5D |
|---|---:|---:|
| <=50 px | 0.3471 | 0.2851 |
| 51-150 px | 0.4792 | 0.4749 |
| 151-300 px | 0.5439 | 0.5491 |
| 301-600 px | 0.6543 | 0.6576 |
| 601-1200 px | 0.7036 | 0.6999 |

Conclusion:

- CT 2.5D es un experimento negativo/neutral, no una nueva mejor variante.
- La informacion volumetrica local no basta con esta configuracion; parece mejorar algo lesiones medias, pero empeora lesiones muy pequenas.
- El mejor modelo CT sigue siendo `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full`.

Siguiente ajuste preparado:

- `notebooks/05g_ct_mixed_context_ablation.ipynb`
- Variante 2D mas conservadora:
  - patch `192x192`,
  - `train_crop_prob = 0.3`,
  - `positive_crop_prob = 0.7`,
  - `pos_weight = 10`,
  - threshold tuning limitado a `0.80` para comparabilidad con el mejor mixed-context actual.

## 8. Resultado de la ablacion mixed-context 2D

Tras ejecutar `notebooks/05g_ct_mixed_context_ablation.ipynb`, la variante 2D mas conservadora pasa a ser el mejor modelo CT individual:

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| `ct_attention_unet_mixed50_patch160_pos80_tversky_pos20_thr_segmentation_full` | 0.5242 | 0.3846 | 0.9959 | 0.80 |

Mejora frente al mixed-context anterior:

- Dice: `+0.0062`.
- IoU: `+0.0096`.
- Pixeles positivos medios predichos: baja de `399.8` a `342.3`, acercandose al objetivo real medio de `229.3`.

Interpretacion:

- La mejora es moderada pero consistente en IoU.
- El modelo nuevo reduce sobresegmentacion respecto al mixed-context anterior.
- Mejora especialmente en lesiones medianas/grandes:
  - `151-300 px`: Dice `0.5439` -> `0.5538`.
  - `301-600 px`: Dice `0.6543` -> `0.6995`.
  - `601-1200 px`: Dice `0.7036` -> `0.7240`.
- En lesiones `51-150 px` empeora (`0.4792` -> `0.4505`), por lo que aun existe margen en lesiones pequenas/medianas.

Decision:

- Adoptar `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_thr080_segmentation_full` como mejor modelo CT individual.
- Mantener el mixed-context anterior como comparador porque ambos modelos son complementarios por estudio y tamano de lesion.

Siguiente paso preparado:

- `notebooks/05h_ct_mixed_ensemble.ipynb`
- Combina probabilidades de los dos mejores modelos mixed-context.
- Selecciona peso del ensemble y threshold exclusivamente en validacion.
- Evalua una vez en test y guarda un summary comparable con el resto de resultados.

## 9. Resultado del ensemble validado

Tras ejecutar `notebooks/05h_ct_mixed_ensemble.ipynb`, el ensemble se convierte en el mejor resultado CT global.

La seleccion se hizo en validacion:

- Peso modelo `mixed50_patch160_pos20`: `0.80`.
- Peso modelo `mixed30_patch192_pos10`: `0.20`.
- Threshold: `0.90`.
- Dice en validacion: `0.5153`.
- IoU en validacion: `0.3906`.

Evaluacion final en test:

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| Ensemble mixed-context validado | 0.5444 | 0.4097 | 0.9972 | 0.90 |
| Mejor modelo individual `mixed30_patch192_pos70_tversky_pos10_thr080` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| Modelo individual anterior `mixed50_patch160_pos80_tversky_pos20_thr` | 0.5242 | 0.3846 | 0.9959 | 0.80 |
| Baseline Attention U-Net | 0.5001 | 0.3648 | 0.9960 | 0.50 |

Mejora del ensemble:

- Frente al mejor modelo individual:
  - Dice: `+0.0140`.
  - IoU: `+0.0156`.
- Frente al baseline Attention U-Net:
  - Dice: `+0.0443`.
  - IoU: `+0.0449`.

Interpretacion:

- El ensemble aprovecha la complementariedad de los dos modelos mixed-context.
- El modelo `mixed50_patch160_pos20` aporta sensibilidad a lesiones mas pequenas.
- El modelo `mixed30_patch192_pos10` aporta una prediccion mas conservadora y reduce sobresegmentacion.
- Como el peso y el threshold se seleccionaron en validacion, el resultado es metodologicamente defendible.
- El resultado de test debe interpretarse con prudencia porque el conjunto CT anotado es pequeno, pero representa la mejor configuracion obtenida hasta ahora.

Decision final recomendada:

- Reportar el ensemble como mejor resultado CT global.
- Reportar `mixed30_patch192_pos70_tversky_pos10_thr080` como mejor modelo CT individual.
- No seguir entrenando variantes CT salvo que se quiera hacer validacion cruzada por estudio.
- Pasar a visualizacion cualitativa, matriz/comparativa final e informe de Fase 2.

## 10. Experimentos adicionales si se quiere seguir explorando

Aunque el ensemble ya es el mejor resultado CT global, se prepararon dos experimentos adicionales para explorar si queda margen:

### 10.1 Entrenamiento con slices negativos

Notebook:

- `notebooks/05i_ct_negative_slice_training.ipynb`

Hipotesis:

- Los modelos actuales entrenan principalmente sobre slices con lesion.
- Incluir slices negativos de los mismos estudios de train puede reducir falsos positivos y sobresegmentacion.

Configuracion:

- Misma variante base que el mejor modelo individual 2D.
- Train con slices positivos y negativos en proporcion `1:1`.
- Val/test principales se mantienen en slices positivos para comparabilidad.
- Threshold tuning hasta `0.95`.

Riesgo:

- Puede volverse demasiado conservador y perder sensibilidad en lesiones pequenas.
- Si empeora, probar `NEGATIVE_RATIO = 0.5`.

### 10.2 Refinamiento de mayor capacidad

Notebook:

- `notebooks/05j_ct_high_capacity_refinement.ipynb`

Hipotesis:

- El mejor modelo individual usa `base_features=16`.
- Subir a `base_features=32` puede mejorar representacion si el modelo estaba limitado por capacidad.

Configuracion:

- Misma receta `mixed30_patch192_pos70_tversky_pos10`.
- `base_features = 32`.
- `batch_size = 4`.
- `epochs = 36`.
- `early_stopping_patience = 8`.
- Threshold tuning hasta `0.95`.

Riesgo:

- Puede sobreajustar por el tamano pequeno del dataset CT.
- Sera mas lento que los experimentos anteriores.

Orden recomendado:

1. Ejecutar primero `05i`, porque prueba una mejora de datos y no solo mas capacidad.
2. Ejecutar `05j` solo si se quiere invertir mas tiempo de computo.
3. Comparar siempre contra:
   - mejor CT global: ensemble Dice `0.5444`, IoU `0.4097`;
   - mejor CT individual: Dice `0.5304`, IoU `0.3942`.

## 11. Resultado del refinamiento de mayor capacidad

Tras ejecutar `notebooks/05j_ct_high_capacity_refinement.ipynb`, la variante de mayor capacidad se convierte en el mejor resultado CT global e individual:

| Experimento | Dice | IoU | Pixel accuracy | Threshold |
|---|---:|---:|---:|---:|
| `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full` | 0.5637 | 0.4305 | 0.9977 | 0.90 |
| Ensemble mixed-context validado | 0.5444 | 0.4097 | 0.9972 | 0.90 |
| Mejor modelo individual anterior `mixed30_patch192_pos70_tversky_pos10_thr080` | 0.5304 | 0.3942 | 0.9965 | 0.80 |
| Baseline Attention U-Net | 0.5001 | 0.3648 | 0.9960 | 0.50 |

Mejoras:

- Frente al ensemble anterior:
  - Dice: `+0.0193`.
  - IoU: `+0.0208`.
- Frente al mejor modelo individual anterior:
  - Dice: `+0.0333`.
  - IoU: `+0.0364`.
- Frente al baseline Attention U-Net:
  - Dice: `+0.0636`.
  - IoU: `+0.0657`.

Interpretacion:

- Aumentar `base_features` de `16` a `32` si aporta mejora real en CT.
- No fue simplemente "entrenar mas": se aumento capacidad, se bajo `batch_size` a `4` y se mantuvo early stopping.
- La validacion llego a Dice alto durante entrenamiento y el threshold seleccionado fue `0.90`, elegido en validacion.
- Se probo un ensemble diagnostico con los modelos anteriores y la seleccion en validacion eligio `100%` del modelo bf32; por tanto, los modelos anteriores ya no aportan mejora al ensemble.

Decision:

- Adoptar `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation_full` como mejor resultado CT final por ahora.
- Mantener el ensemble anterior como experimento intermedio, no como resultado final.
- No crear otro ensemble con bf32 porque la validacion selecciona el modelo bf32 puro.

Siguiente posible experimento, solo si se quiere seguir explorando:

- Combinar la receta bf32 con slices negativos de train.
- Hipotesis: reducir falsos positivos sin perder la capacidad nueva.
- Riesgo: puede volverse demasiado conservador y empeorar lesiones pequenas.
- `notebooks/05i_ct_negative_slice_training.ipynb` queda ajustado para usar esta receta bf32.
