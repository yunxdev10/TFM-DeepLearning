# Informe de ajuste para segmentacion CT

Fecha: 2026-05-14

## 1. Lectura de resultados actuales

Resultados full disponibles:

| Experimento | Dataset | Arquitectura | Dice | IoU | Pixel accuracy |
|---|---|---|---:|---:|---:|
| `cxr_attention_unet_segmentation_full` | CXR | Attention U-Net | 0.9853 | 0.9715 | 0.9935 |
| `cxr_unet_segmentation_full` | CXR | U-Net | 0.9853 | 0.9713 | 0.9935 |
| `ct_attention_unet_segmentation_full` | CT | Attention U-Net | 0.5001 | 0.3648 | 0.9960 |
| `ct_unet_segmentation_full` | CT | U-Net | 0.4790 | 0.3522 | 0.9971 |

Interpretacion:

- CXR funciona muy bien. Dice ≈ 0.985 e IoU ≈ 0.971 son resultados fuertes para segmentacion pulmonar.
- CT queda claramente por debajo. Attention U-Net mejora ligeramente a U-Net, pero ambas quedan alrededor de Dice 0.48-0.50.
- El `pixel_accuracy` de CT no debe usarse como metrica principal. Es alto porque casi todos los pixeles son fondo.

## 2. Diagnostico del problema CT

Se estimo la proporcion de pixeles positivos en las mascaras CT:

| Split | Slices | Fraccion de lesion | Pos weight bruto |
|---|---:|---:|---:|
| Train | 508 | 0.005717 | 173.9 |
| Val | 86 | 0.003620 | 275.3 |
| Test | 110 | 0.003499 | 284.8 |

Esto significa que la lesion ocupa aproximadamente entre 0.35% y 0.57% de la imagen. Es un desbalance extremo a nivel de pixel.

Consecuencias:

- Un modelo puede obtener `pixel_accuracy` muy alta prediciendo casi todo fondo.
- Dice e IoU son las metricas relevantes.
- La funcion Dice+BCE baseline puede no penalizar suficientemente los falsos negativos de lesion.
- CT tiene solo 50 estudios anotados, por lo que la variabilidad entre train/val/test puede ser alta.

## 3. Barrido de umbral

Se evaluo si cambiar el umbral de binarizacion mejoraba CT sin reentrenar.

### U-Net CT

| Umbral | Dice | IoU | Pixeles predichos medios | Pixeles reales medios |
|---:|---:|---:|---:|---:|
| 0.3 | 0.1739 | 0.1015 | 1591.2 | 229.3 |
| 0.4 | 0.4933 | 0.3585 | 322.2 | 229.3 |
| 0.5 | 0.4790 | 0.3522 | 251.9 | 229.3 |
| 0.6 | 0.4471 | 0.3304 | 197.2 | 229.3 |

Mejor umbral observado: 0.4.

### Attention U-Net CT

| Umbral | Dice | IoU | Pixeles predichos medios | Pixeles reales medios |
|---:|---:|---:|---:|---:|
| 0.3 | 0.4372 | 0.3031 | 634.9 | 229.3 |
| 0.4 | 0.4910 | 0.3533 | 464.5 | 229.3 |
| 0.5 | 0.5001 | 0.3648 | 373.2 | 229.3 |
| 0.6 | 0.4856 | 0.3568 | 304.6 | 229.3 |
| 0.7 | 0.4564 | 0.3380 | 240.4 | 229.3 |

Mejor umbral observado: 0.5.

Conclusion:

- Ajustar el umbral por si solo no resuelve CT.
- En U-Net aporta una mejora pequena, de 0.4790 a 0.4933.
- En Attention U-Net el umbral actual 0.5 ya es adecuado.

## 4. Cambios implementados para mejorar experimentacion

Se amplio `src/training/segmentation_experiment.py` para permitir:

- variantes con nombre propio (`variant_name`) sin sobrescribir resultados baseline,
- `weighted_dice_bce`,
- `weighted_tversky_bce`,
- `pos_weight` para compensar desbalance foreground/background,
- parametros Tversky `alpha` y `beta`,
- optimizacion de umbral en validacion (`optimize_threshold=True`),
- guardado del umbral, loss y variante en checkpoint y summary.

Tambien se actualizo `notebooks/05_segmentation.ipynb` con un bloque de hiperparametros preparado para CT.

Ejemplo de variante recomendada:

```python
HYPERPARAMETER_OVERRIDES = {
    'ct': {
        'variant_name': 'tversky_pos30_thr',
        'loss_name': 'weighted_tversky_bce',
        'bce_weight': 0.2,
        'dice_weight': 0.8,
        'pos_weight': 30.0,
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7,
        'optimize_threshold': True,
        'early_stopping_patience': 10,
    },
}
```

## 5. Experimentos descartados durante ajuste

Se probaron dos arranques de entrenamiento CT con Tversky ponderado desde consola, pero se interrumpieron porque:

- la variante con `base_features=32` era demasiado lenta para iteracion rapida,
- las primeras validaciones eran peores que el baseline,
- la variante `base_features=16` seguia siendo lenta en este entorno y no llego a completar una epoca.

No se guardaron artefactos parciales de esas variantes. Los resultados baseline full siguen intactos.

## 6. Recomendacion metodologica

Para mejorar CT de forma defendible:

1. Mantener CXR como resultado fuerte de Fase 2.
2. No optimizar contra `pixel_accuracy`; usar Dice e IoU.
3. Probar variantes CT con:
   - `weighted_tversky_bce`,
   - `pos_weight` acotado entre 20 y 50,
   - `beta > alpha` para penalizar falsos negativos,
   - umbral optimizado en validacion.
4. Ejecutar primero una prueba a menor coste:
   - `base_features=16`,
   - `batch_size=8` o `16`,
   - `epochs=10-15`,
   - `early_stopping_patience=4-5`.
5. Si mejora Dice de validacion, repetir la mejor variante con:
   - `base_features=32`,
   - `epochs=30-50`,
   - `early_stopping_patience=10`.

## 7. Interpretacion para la memoria si CT no mejora

Si CT se mantiene alrededor de Dice 0.50, sigue siendo un resultado defendible:

- La segmentacion de lesion COVID en CT es mucho mas dificil que la segmentacion pulmonar CXR.
- El dataset CT anotado tiene solo 50 estudios.
- Las lesiones ocupan menos del 1% de los pixeles.
- La variabilidad de severidad y localizacion de lesiones es alta.
- La metrica adecuada para discutir CT es Dice/IoU, no pixel accuracy.

Texto posible:

> La segmentacion pulmonar en CXR alcanzo un rendimiento muy alto con U-Net y Attention U-Net, con Dice cercano a 0.985. En cambio, la segmentacion de lesiones en CT mostro un rendimiento moderado, alrededor de Dice 0.50. Esta diferencia no se debe a un fallo del pipeline, sino a la naturaleza mucho mas exigente de la tarea CT: las mascaras de infeccion ocupan menos del 1% de los pixeles y solo se dispone de 50 estudios anotados. Por ello, CT debe interpretarse como una tarea de segmentacion de lesion pequena y altamente desbalanceada, donde Dice e IoU son mucho mas informativas que pixel accuracy.
