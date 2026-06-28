# Plan TFM: Deep Learning Multi-modal para Diagnóstico y Explicabilidad de COVID-19

## TL;DR
TFM que compara arquitecturas de deep learning (ResNet-50, DenseNet-121, EfficientNet-B0) sobre dos modalidades de imagen médica: radiografías de tórax (CXR, 4 clases) y tomografía computarizada (CT, 4 grupos tras fusionar CT-3 y CT-4 en CT-3+). Incluye segmentación pulmonar en CXR, segmentación de infección/lesión en CT, análisis de explicabilidad con Grad-CAM y estrategias de manejo de desbalanceo de clases. Marco unificado con código modular en PyTorch.

**Título propuesto:** *"Clasificación, Segmentación y Explicabilidad de COVID-19 en Imagen Médica: Un Estudio Comparativo Multi-modal con Deep Learning"*

**Plazo:** 20 abril – 31 agosto 2026 (~19 semanas)
**Idioma:** Español
**Recursos:** GPU local, nivel intermedio PyTorch

---

## Preguntas de Investigación

**RQ Principal:** ¿En qué medida las arquitecturas de deep learning con transfer learning logran una clasificación fiable y explicable de COVID-19 y patologías relacionadas en CXR y CT, considerando las diferencias clínicas, de etiquetas y de anotaciones disponibles entre modalidades?

- **RQ1 (Transfer Learning):** ¿Qué arquitectura (ResNet-50, DenseNet-121, EfficientNet-B0) ofrece mejor balance accuracy/sensibilidad/eficiencia en cada modalidad?
- **RQ2 (Cross-Modal):** ¿Cómo se compara el rendimiento entre CXR (4 clases: COVID, Lung Opacity, Normal, Viral Pneumonia) y CT (4 grupos de severidad: CT-0, CT-1, CT-2, CT-3+) usando arquitecturas y metodología equivalentes?
- **RQ3 (Explicabilidad):** ¿En qué medida Grad-CAM produce explicaciones coherentes con las regiones anatómicas o patológicas disponibles: región pulmonar en CXR y lesiones/infección en CT?
- **RQ4 (Segmentación):** ¿Qué variante de U-Net (vanilla, Attention, ResUNet) logra mejor segmentación en las tareas disponibles: segmentación pulmonar en CXR y segmentación de lesiones COVID en CT?
- **RQ5 (Desbalanceo):** ¿Qué estrategia (focal loss, class weights, oversampling, augmentation) mejora más la sensibilidad en clases minoritarias?

---

## FASE 0 — Setup y Refactorización (Semanas 1-2: 20 abril – 3 mayo)

### Paso 0.1: Estructura del proyecto
- Crear estructura modular:
  ```
  src/
    data/
      datasets.py        # LungCTDataset, CXRDataset, SegmentationDataset
      transforms.py       # Augmentaciones y preprocesamiento
      preprocessing.py    # NIfTI → PNG, splits
    models/
      classifiers.py      # ResNet50, DenseNet121, EfficientNetB0 wrappers
      segmentation.py     # UNet, AttentionUNet
    training/
      trainer.py          # Loop de entrenamiento genérico
      losses.py           # FocalLoss, DiceLoss, combinadas
    evaluation/
      metrics.py          # Accuracy, F1, AUC-ROC, Dice, IoU
      explainability.py   # GradCAM, SHAP, LIME wrappers
      visualization.py    # Plots, confusion matrices, curvas ROC
    config.py             # Hiperparámetros centralizados
  ```
- Migrar código funcional de `01_data_loading.ipynb` a módulos `src/`
- Corregir rutas hardcodeadas (`/Users/alex/...` → rutas relativas con `pathlib`)
- Configurar `requirements.txt` con versiones fijas

### Paso 0.2: Análisis Exploratorio de Datos (EDA)
- Notebook `notebooks/00_eda.ipynb`:
  - Distribución de clases en ambos datasets (gráficos de barras)
  - Visualización de ejemplos por clase (grid de imágenes)
  - Estadísticas de intensidad (histogramas, media, std)
  - Para MosMedData: distribución de slices por estudio, variabilidad de tamaño de volumen
  - Para Kaggle: análisis de las máscaras de segmentación (% de área pulmonar, distribución)
  - Tabla resumen comparativa de ambos datasets

### Paso 0.3: Preprocesamiento robusto
- **MosMedData CT:**
  - Mantener pipeline existente: NIfTI → slices centrales (20-80%) → resize 256×256 → PNG
  - Añadir: windowing HU [-1000, 400] para mejor contraste pulmonar
  - Regenerar labels.csv con metadata adicional (study_id, slice_index, total_slices)
  - Agrupar CT-3 y CT-4 en una sola clase "CT-3+" (solo 47 estudios combinados → más viable)
- **Kaggle CXR:**
  - Resize a 224×224 (estándar ImageNet)
  - Split estratificado 70/15/15 (train/val/test) con seed fijo
  - Separar imágenes y máscaras de segmentación

**Archivos a modificar/crear:**
- `notebooks/01_data_loading.ipynb` → refactorizar y extraer a `src/data/`
- `src/data/preprocessing.py` — nuevo
- `src/data/datasets.py` — nuevo
- `src/config.py` — nuevo
- `notebooks/00_eda.ipynb` — nuevo

**Verificación Fase 0:**
- [x] Datasets cargados correctamente con DataLoaders (CXR: batch 4×3×224×224; CT: batch 4×1×256×256)
- [x] EDA notebook ejecutable de inicio a fin sin errores (`notebooks/00_eda.ipynb`)
- [x] Estructura `src/` importable desde notebooks (`from src.data import ...`)
- [x] CT preprocesado a slices 2D con metadata (`27.781` PNGs + `labels_metadata.csv`)

---

## FASE 1 — Clasificación con Transfer Learning (Semanas 3-9: 4 mayo – 21 junio)

### Paso 1.1: Implementar modelos de clasificación
- Wrappers en `src/models/classifiers.py` para:
  - **ResNet-50** (pretrained ImageNet): modificar fc final → num_classes
  - **DenseNet-121** (pretrained ImageNet): modificar classifier → num_classes
  - **EfficientNet-B0** (pretrained ImageNet): modificar _fc → num_classes
- Para CT (1 canal grayscale): replicar canal 1→3 o modificar primera conv
- Fine-tuning strategy: congelar backbone → entrenar cabeza 5 epochs → descongelar últimas capas → entrenar todo 15 epochs más (lr reducido)

### Paso 1.2: Data Augmentation
- En `src/data/transforms.py`:
  - Train: RandomHorizontalFlip, RandomRotation(15), RandomAffine, ColorJitter (suave), Normalize con media/std de ImageNet
  - Val/Test: solo Resize + Normalize
  - Para CT: augmentaciones más conservadoras (la anatomía importa)

### Paso 1.3: Manejo del desbalanceo (RQ5)
- En `src/training/losses.py`:
  - Weighted CrossEntropy (pesos inversamente proporcionales a frecuencia de clase)
  - Focal Loss (gamma=2, alpha por clase)
- En `src/data/datasets.py`:
  - WeightedRandomSampler para oversampling de clases minoritarias
- Ejecutar **ablation study**: cada estrategia vs baseline sin balanceo

### Paso 1.4: Entrenamiento — Dataset CXR (Kaggle)
- Notebook `notebooks/02_cxr_classification.ipynb`:
  - 3 arquitecturas × {sin balanceo, weighted CE, focal loss} = 9 experimentos
  - 20 epochs, early stopping patience=5, ReduceLROnPlateau
  - Guardar mejores modelos en `models/cxr/`
  - Log: train/val loss, val accuracy por epoch

### Paso 1.5: Entrenamiento — Dataset CT (MosMedData)
- Notebook `notebooks/03_ct_classification.ipynb`:
  - Mismos 9 experimentos que CXR
  - 4 clases (CT-0, CT-1, CT-2, CT-3+) tras merge
  - Guardar en `models/ct/`

### Paso 1.6: Evaluación comparativa (RQ1, RQ2)
- Notebook `notebooks/04_classification_results.ipynb`:
  - Classification report por modelo × dataset × estrategia de balanceo
  - Matrices de confusión lado a lado
  - Curvas ROC multiclase (one-vs-rest) con AUC
  - Tabla resumen: Accuracy, F1-macro, F1-weighted, AUC-macro
  - **Análisis cross-modal**: tabla comparativa CXR vs CT para cada arquitectura
  - Análisis de errores: qué clases se confunden más y por qué
  - Test de significancia estadística (McNemar o paired t-test sobre folds)

**Archivos a crear:**
- `src/models/classifiers.py`
- `src/data/transforms.py`
- `src/training/trainer.py`
- `src/training/losses.py`
- `src/evaluation/metrics.py`
- `notebooks/02_cxr_classification.ipynb`
- `notebooks/03_ct_classification.ipynb`
- `notebooks/04_classification_results.ipynb`

**Verificación Fase 1:**
- [ ] Los 3 modelos convergen en ambos datasets (val loss decrece)
- [ ] Accuracy CXR ≥ 90% con mejor modelo (la literatura reporta 93-97%)
- [ ] Accuracy CT ≥ 85% con mejor modelo (baseline actual 74% con SimpleCNN)
- [ ] Tablas comparativas generadas con todos los resultados

---

## FASE 2 — Segmentación Pulmonar y de Lesiones (Semanas 10-11: 22 junio – 5 julio)

### Paso 2.1: Implementar modelos de segmentación
- En `src/models/segmentation.py`:
  - **U-Net vanilla**: encoder-decoder con skip connections
  - **Attention U-Net**: con attention gates en skip connections
  - Usar `segmentation_models_pytorch` (librería SMP) para implementaciones robustas
  - Encoder: ResNet-34 pretrained (compartir conocimiento con clasificación)

### Paso 2.2: Dataset de segmentación
- **Kaggle CXR:** 21,165 imágenes con máscaras de pulmón → segmentación de región pulmonar
- **MosMedData CT:** 50 máscaras de infección (study_0255–0304) → segmentación de lesión COVID
- Augmentaciones geométricas sincronizadas imagen↔máscara (flip, rotate, scale)

### Paso 2.3: Entrenamiento de segmentación
- Notebook `notebooks/05_segmentation.ipynb`:
  - Loss: Dice Loss + BCE (combinada)
  - Métricas: Dice coefficient, IoU (Jaccard), Pixel Accuracy
  - Entrenar U-Net y Attention U-Net en ambos datasets
  - 30 epochs, early stopping

### Paso 2.4: Evaluación de segmentación (RQ4)
- Visualización: imagen original → ground truth → predicción (3 columnas)
- Tabla comparativa: U-Net vs Attention U-Net × CXR vs CT
- Análisis cualitativo separado por modalidad:
  - CXR: ¿la segmentación delimita de forma estable el campo pulmonar?
  - CT: ¿la segmentación captura regiones de infección compatibles con lesiones COVID?

**Archivos a crear:**
- `src/models/segmentation.py`
- `notebooks/05_segmentation.ipynb`

**Verificación Fase 2:**
- [ ] Dice score CXR (pulmón) ≥ 0.90 (es relativamente fácil)
- [ ] Dice score CT (lesión) ≥ 0.60 (más difícil, solo 50 muestras anotadas)
- [ ] Visualizaciones de segmentación coherentes clínicamente

---

## FASE 3 — Explicabilidad (Semanas 12-13: 6-19 julio)

*Paralela parcialmente con el cierre de Fase 2 si los modelos finales ya están guardados.*

### Paso 3.1: Implementar métodos XAI
- En `src/evaluation/explainability.py`:
  - **Grad-CAM**: implementado en PyTorch sobre la última capa convolucional.
  - **LIME/SHAP**: quedan como extensión opcional no incluida en el alcance final para evitar dependencias adicionales y coste computacional que no aportan a la conclusión principal.

### Paso 3.2: Generar visualizaciones XAI
- Notebook `notebooks/06_explainability.ipynb`:
  - Seleccionar imágenes representativas por clase (correctas e incorrectas).
  - Aplicar Grad-CAM al mejor modelo de cada modalidad y al modelo CT alternativo de mayor accuracy.
  - Grid cualitativo: Original | Máscara | Grad-CAM | Saliencia binaria | Máscara vs saliencia.

### Paso 3.3: Evaluación cuantitativa de XAI (RQ3) — **Contribución novedosa**
- Calcular **IoU entre mapa de saliencia y máscara disponible**, con interpretación diferenciada:
  - Binarizar saliencia (threshold) → comparar con máscara
  - CXR: saliencia vs máscara pulmonar → mide si el modelo atiende al área anatómica relevante, no a lesiones COVID
  - CT: saliencia vs máscara de infección → mide alineación con regiones patológicas anotadas
  - Reportar IoU medio por método XAI y por modalidad
- Interpretación clave: en CXR, una alta superposición con pulmón no demuestra localización de lesión; solo descarta parcialmente atención fuera del campo pulmonar.
- Comparar si Grad-CAM se alinea de forma suficiente con la región disponible para cada modalidad.

### Paso 3.4: Análisis cualitativo
- Seleccionar casos interesantes:
  - Predicción correcta con buena explicación
  - Predicción correcta pero explicación en zona irrelevante (modelo "correcto por las razones equivocadas")
  - Predicción incorrecta con explicación que revela el error
- Discusión clínica: ¿los mapas de atención corresponden a patrones radiológicos conocidos?

**Archivos a crear:**
- `src/evaluation/explainability.py`
- `notebooks/06_explainability.ipynb`

**Verificación Fase 3:**
- [x] Grad-CAM genera visualizaciones coherentes para CXR y CT
- [x] IoU saliencia-máscara calculado separando CXR-pulmón y CT-lesión
- [x] Tabla cuantitativa Grad-CAM generada para CXR y CT

---

## FASE 4 — Integración, Análisis y Redacción (Semanas 14-19: 20 julio – 31 agosto)

### Paso 4.1: Experimentos finales y consolidación (Semana 14)
- Notebook `notebooks/07_final_analysis.ipynb`:
  - Tabla maestra con TODOS los resultados (clasificación + segmentación + XAI)
  - Análisis estadístico: intervalos de confianza, tests de significancia
  - Gráficos de resumen publicables (estilo paper)
  - Responder explícitamente cada RQ con evidencia
- Notebook `notebooks/08_calibration_analysis.ipynb`:
  - Análisis de calibración probabilística sin reentrenar modelos
  - Reliability diagrams, ECE, Brier score y errores de alta confianza
  - Discusión de si la confianza de los clasificadores refleja su fiabilidad real

### Paso 4.2: Redacción de la memoria (Semanas 15-19)
Estructura propuesta:

1. **Introducción** (~5 páginas)
   - Contexto: COVID-19 y diagnóstico por imagen
   - Motivación y relevancia
   - Objetivos y preguntas de investigación
   - Estructura del documento

2. **Estado del Arte** (~15 páginas)
   - 2.1 COVID-19 y manifestaciones radiológicas
   - 2.2 Deep learning en imagen médica
   - 2.3 Transfer learning: ResNet, DenseNet, EfficientNet
   - 2.4 Segmentación: U-Net y variantes
   - 2.5 Explicabilidad en IA médica: Grad-CAM, SHAP, LIME
   - 2.6 Manejo de desbalanceo de clases
   - 2.7 Trabajos relacionados y gaps identificados

3. **Metodología** (~15 páginas)
   - 3.1 Datasets: descripción, estadísticas, preprocesamiento
   - 3.2 Arquitecturas de clasificación
   - 3.3 Arquitecturas de segmentación
   - 3.4 Estrategia de transfer learning y fine-tuning
   - 3.5 Manejo del desbalanceo
   - 3.6 Métodos de explicabilidad
   - 3.7 Calibración e incertidumbre probabilística
   - 3.8 Métricas de evaluación
   - 3.9 Diseño experimental

4. **Resultados** (~20 páginas)
   - 4.1 EDA y análisis de datos
   - 4.2 Clasificación CXR: comparativa de arquitecturas
   - 4.3 Clasificación CT: comparativa de arquitecturas
   - 4.4 Análisis cross-modal
   - 4.5 Impacto del manejo de desbalanceo
   - 4.6 Segmentación pulmonar en CXR y segmentación de lesiones en CT
   - 4.7 Análisis de explicabilidad
   - 4.8 Evaluación cuantitativa XAI vs máscaras disponibles
   - 4.9 Calibración de los clasificadores y errores de alta confianza

5. **Discusión** (~8 páginas)
   - Respuestas a las RQs
   - Comparación con la literatura
   - Limitaciones
   - Implicaciones clínicas

6. **Conclusiones y Trabajo Futuro** (~3 páginas)

7. **Bibliografía** (~50-70 referencias)

8. **Anexos**: código relevante, resultados adicionales

### Paso 4.3: Preparación de la defensa (última semana de agosto)
- Presentación PowerPoint/Beamer (~20-25 slides)
- Ensayo de la presentación (15-20 min)

**Verificación Fase 4:**
- [ ] Memoria completa con todas las secciones
- [ ] Todos los notebooks ejecutables de inicio a fin
- [ ] Código subido a repositorio (GitHub) con README
- [ ] Presentación preparada

---

## Cronograma Semanal

| Semana | Fechas | Fase | Entregable |
|--------|--------|------|------------|
| 1 | 20-26 abr | F0: Setup | Proyecto modular inicial |
| 2 | 27 abr-3 may | F0: EDA + preprocesamiento | EDA notebook, splits CXR, pipeline CT validado |
| 3 | 4-10 may | F1: Preparación clasificación | DataLoaders verificados, baseline listo |
| 4 | 11-17 may | F1: Clasificación CXR baseline | Primer modelo CXR entrenado y evaluado |
| 5 | 18-24 may | F1: Clasificación CXR completa | ResNet/DenseNet/EfficientNet en CXR |
| 6 | 25-31 may | F1: CT preprocessing + baseline | Slices CT generados, primer modelo CT |
| 7 | 1-7 jun | F1: Clasificación CT completa | ResNet/DenseNet/EfficientNet en CT |
| 8 | 8-14 jun | F1: Desbalanceo | Ablation baseline vs weighted CE vs focal/oversampling |
| 9 | 15-21 jun | F1: Resultados clasificación | Tablas, matrices, ROC, análisis de errores |
| 10 | 22-28 jun | F2: Segmentación setup | Dataset segmentación, U-Net/Attention U-Net |
| 11 | 29 jun-5 jul | F2: Segmentación entrenamiento/eval | Métricas Dice/IoU y visualizaciones |
| 12 | 6-12 jul | F3: Grad-CAM | Mapas Grad-CAM para mejores modelos |
| 13 | 13-19 jul | F3: Evaluación XAI | Grad-CAM y saliencia vs máscaras disponibles |
| 14 | 20-26 jul | F4: Análisis final | Notebook final, estadística, respuesta a RQs |
| 15 | 27 jul-2 ago | F4: Redacción | Introducción + Estado del Arte |
| 16 | 3-9 ago | F4: Redacción | Metodología + Resultados |
| 17 | 10-16 ago | F4: Redacción | Discusión + Conclusiones |
| 18 | 17-23 ago | F4: Revisión | Memoria revisada, repo y anexos cerrados |
| 19 | 24-31 ago | F4: Defensa | Memoria final + presentación |

---

## Referencias Clave a Citar

### Datasets (cita obligatoria)
- Morozov et al., 2020 — MosMedData (arXiv:2005.06465) — 427 citas
- Rahman et al., 2021 — COVID-19 Radiography Dataset (Kaggle)

### Transfer Learning
- Gifani et al., 2021 — Ensemble transfer learning CT (207 citas)
- Showkat & Qureshi, 2022 — ResNet variants CXR (143 citas)
- Kumar et al., 2023 — EfficientNet + GoogLeNet (117 citas)

### Explicabilidad
- Bhandari et al., 2022 — SHAP + LIME + Grad-CAM framework (214 citas)
- Rajpoot et al., 2024 — Ensemble CNN + XAI comparativo
- Khan et al., 2022 — XAI framework COVID CXR (52 citas)

### Segmentación
- Saood & Hatem, 2021 — U-Net vs SegNet COVID (338 citas)
- Sánchez Ocampo & Reyes, 2024 — U-Net variants COVID

### Surveys
- Al-qaness et al., 2024 — CXR survey (88 citas)
- Khan et al., 2024 — DL COVID survey (99 citas)

---

## Decisiones y Scope

**Incluido:**
- Clasificación multi-clase con 3 arquitecturas × 2 datasets × 3 estrategias balanceo
- Segmentación con U-Net y Attention U-Net en dos tareas distintas: pulmón en CXR y lesión/infección en CT
- Explicabilidad con Grad-CAM + evaluación cuantitativa contra máscaras disponibles
- Análisis cross-modal controlado, interpretado como comparación metodológica y no como equivalencia clínica directa entre etiquetas CXR y severidades CT
- Manejo del desbalanceo de clases

**Excluido deliberadamente (para mantener alcance realista hasta finales de agosto):**
- Enfoques 3D volumétricos (requieren mucha más GPU y tiempo)
- LIME/SHAP como experimentos principales (se dejan como extensión opcional por dependencias, coste computacional y alcance)
- Multi-task learning (clasificación + segmentación simultánea) — mencionado como trabajo futuro
- Modelos generativos (GANs) para augmentación
- Validación clínica con radiólogos
- Deployment/aplicación web
- Afirmar localización de lesiones en CXR usando máscaras pulmonares; esas máscaras solo permiten evaluar atención dentro/fuera del campo pulmonar

**Decisiones técnicas:**
- Merge CT-3 + CT-4 en una sola clase (CT-4 solo tiene 2 estudios → inviable estadísticamente)
- Usar `segmentation_models_pytorch` para U-Net (robusto y rápido de implementar)
- Usar `pytorch-grad-cam`, `lime`, `shap` como librerías XAI (no reimplementar)
- Input size: 224×224 para CXR (estándar ImageNet), 256×256 para CT (ya preprocesado)
