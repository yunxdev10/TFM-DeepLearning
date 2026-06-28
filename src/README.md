# Codigo fuente del TFM

Esta carpeta contiene la logica reutilizable del proyecto. Los notebooks y scripts llaman a estos modulos para evitar duplicar codigo experimental.

## Estructura

- `config.py`: rutas del proyecto, tamanos de imagen, hiperparametros base y semilla reproducible.
- `data/`: preparacion de datos, particiones, datasets de PyTorch, transformaciones y seleccion de slices CT.
- `models/`: arquitecturas de clasificacion y segmentacion utilizadas en los experimentos.
- `training/`: bucles de entrenamiento, funciones de perdida y ejecucion de experimentos de clasificacion/segmentacion.
- `evaluation/`: metricas, explicabilidad Grad-CAM y postprocesado de segmentacion.

## Criterio de uso

Los notebooks deben contener la narrativa experimental y llamar a funciones de `src` o `scripts`. La logica repetible, como preparar splits, entrenar modelos o calcular metricas, debe vivir aqui para mantener el TFM reproducible.
