# Plan TFM: Deep Learning Multi-modal para Diagnóstico y Explicabilidad de COVID-19

## TL;DR
TFM que compara arquitecturas de deep learning (ResNet-50, DenseNet-121, EfficientNet-B0) sobre dos modalidades de imagen médica (radiografías de tórax y TC), incluyendo segmentación de lesiones con U-Net, análisis de explicabilidad (Grad-CAM, SHAP, LIME) y estrategias de manejo de desbalanceo de clases. Marco unificado con código modular en PyTorch.

**Título propuesto:** *"Clasificación, Segmentación y Explicabilidad de COVID-19 en Imagen Médica: Un Estudio Comparativo Multi-modal con Deep Learning"*

**Plazo:** 20 abril – 30 junio 2026 (~10 semanas)
**Idioma:** Español
**Recursos:** GPU local, nivel intermedio PyTorch

---

## Preguntas de Investigación

**RQ Principal:** ¿En qué medida las arquitecturas de deep learning con transfer learning logran una clasificación fiable y explicable de COVID-19 en radiografías y TC, y cómo se compara su rendimiento entre modalidades?

- **RQ1 (Transfer Learning):** ¿Qué arquitectura (ResNet-50, DenseNet-121, EfficientNet-B0) ofrece mejor balance accuracy/sensibilidad/eficiencia en cada modalidad?
- **RQ2 (Cross-Modal):** ¿Cómo se compara el rendimiento entre CXR (4 clases) y CT (5 severidades) con arquitecturas y metodología equivalentes?
- **RQ3 (Explicabilidad):** ¿Cuál método XAI (Grad-CAM, SHAP, LIME) produce visualizaciones más alineadas con regiones clínicamente relevantes?
- **RQ4 (Segmentación):** ¿Qué variante de U-Net (vanilla, Attention, ResUNet) logra mejor segmentación de lesiones COVID?
- **RQ5 (Desbalanceo):** ¿Qué estrategia (focal loss, class weights, oversampling, augmentation) mejora más la sensibilidad en clases minoritarias?

---

## FASE 0 — Setup y Refactorización (Semana 1: 20-27 abril)

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
- [ ] Datasets cargados correctamente con DataLoaders (print shapes y labels de 1 batch)
- [ ] EDA notebook ejecutable de inicio a fin sin errores
- [ ] Estructura `src/` importable desde notebooks (`from src.data import ...`)

---

## FASE 1 — Clasificación con Transfer Learning (Semanas 2-3: 28 abril – 11 mayo)

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

## FASE 2 — Segmentación de Lesiones (Semanas 4-5: 12-25 mayo)

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
- Análisis cualitativo: ¿las segmentaciones capturan las zonas de opacidad en vidrio esmerilado?

**Archivos a crear:**
- `src/models/segmentation.py`
- `notebooks/05_segmentation.ipynb`

**Verificación Fase 2:**
- [ ] Dice score CXR (pulmón) ≥ 0.90 (es relativamente fácil)
- [ ] Dice score CT (lesión) ≥ 0.60 (más difícil, solo 50 muestras anotadas)
- [ ] Visualizaciones de segmentación coherentes clínicamente

---

## FASE 3 — Explicabilidad (Semanas 5-6: 25 mayo – 8 junio)

*Paralela parcialmente con final de Fase 2*

### Paso 3.1: Implementar métodos XAI
- En `src/evaluation/explainability.py`:
  - **Grad-CAM**: usando `pytorch-grad-cam` sobre última capa convolucional
  - **LIME**: usando `lime` (LimeImageExplainer) con superpixels
  - **SHAP**: usando `shap` (DeepExplainer o GradientExplainer)

### Paso 3.2: Generar visualizaciones XAI
- Notebook `notebooks/06_explainability.ipynb`:
  - Seleccionar N=50 imágenes representativas por clase (correctas e incorrectas)
  - Aplicar Grad-CAM, LIME, SHAP al mejor modelo de cada modalidad
  - Grid comparativo: Original | Grad-CAM | LIME | SHAP (por fila)

### Paso 3.3: Evaluación cuantitativa de XAI (RQ3) — **Contribución novedosa**
- Calcular **IoU entre mapa de saliencia y máscara de segmentación ground-truth**:
  - Binarizar saliencia (threshold) → comparar con máscara
  - Reportar IoU medio por método XAI
- Esto mide si el modelo "mira donde debe" → alineación clínica
- Comparar: ¿Grad-CAM, SHAP o LIME se alinean mejor con las lesiones reales?

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
- [ ] Los 3 métodos XAI generan visualizaciones coherentes
- [ ] IoU saliencia-máscara calculado para al menos 100 imágenes
- [ ] Tabla comparativa cuantitativa de los 3 métodos

---

## FASE 4 — Integración, Análisis y Redacción (Semanas 7-10: 9-30 junio)

### Paso 4.1: Experimentos finales y consolidación (Semana 7)
- Notebook `notebooks/07_final_analysis.ipynb`:
  - Tabla maestra con TODOS los resultados (clasificación + segmentación + XAI)
  - Análisis estadístico: intervalos de confianza, tests de significancia
  - Gráficos de resumen publicables (estilo paper)
  - Responder explícitamente cada RQ con evidencia

### Paso 4.2: Redacción de la memoria (Semanas 7-10)
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
   - 3.7 Métricas de evaluación
   - 3.8 Diseño experimental

4. **Resultados** (~20 páginas)
   - 4.1 EDA y análisis de datos
   - 4.2 Clasificación CXR: comparativa de arquitecturas
   - 4.3 Clasificación CT: comparativa de arquitecturas
   - 4.4 Análisis cross-modal
   - 4.5 Impacto del manejo de desbalanceo
   - 4.6 Segmentación de lesiones
   - 4.7 Análisis de explicabilidad
   - 4.8 Evaluación cuantitativa XAI vs segmentación

5. **Discusión** (~8 páginas)
   - Respuestas a las RQs
   - Comparación con la literatura
   - Limitaciones
   - Implicaciones clínicas

6. **Conclusiones y Trabajo Futuro** (~3 páginas)

7. **Bibliografía** (~50-70 referencias)

8. **Anexos**: código relevante, resultados adicionales

### Paso 4.3: Preparación de la defensa (última semana de junio)
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
| 1 | 20-27 abr | F0: Setup + EDA | Proyecto modular, EDA notebook |
| 2 | 28 abr-4 may | F1: Clasificación CXR | Modelos CXR entrenados |
| 3 | 5-11 may | F1: Clasificación CT + Resultados | Modelos CT, tablas comparativas |
| 4 | 12-18 may | F2: Segmentación | U-Net entrenados |
| 5 | 19-25 may | F2+F3: Segmentación eval + XAI inicio | Resultados segmentación, Grad-CAM |
| 6 | 26 may-1 jun | F3: Explicabilidad completa | SHAP, LIME, evaluación cuantitativa |
| 7 | 2-8 jun | F4: Análisis final + Redacción inicio | Notebook final, intro+SOTA |
| 8 | 9-15 jun | F4: Redacción | Metodología + Resultados |
| 9 | 16-22 jun | F4: Redacción | Discusión + Conclusiones |
| 10 | 23-30 jun | F4: Revisión + Defensa | Memoria final + presentación |

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
- Segmentación con U-Net y Attention U-Net × 2 datasets
- Explicabilidad con Grad-CAM, SHAP, LIME + evaluación cuantitativa
- Análisis cross-modal controlado
- Manejo del desbalanceo de clases

**Excluido deliberadamente (para mantener alcance realista en 10 semanas):**
- Enfoques 3D volumétricos (requieren mucha más GPU y tiempo)
- Multi-task learning (clasificación + segmentación simultánea) — mencionado como trabajo futuro
- Modelos generativos (GANs) para augmentación
- Validación clínica con radiólogos
- Deployment/aplicación web

**Decisiones técnicas:**
- Merge CT-3 + CT-4 en una sola clase (CT-4 solo tiene 2 estudios → inviable estadísticamente)
- Usar `segmentation_models_pytorch` para U-Net (robusto y rápido de implementar)
- Usar `pytorch-grad-cam`, `lime`, `shap` como librerías XAI (no reimplementar)
- Input size: 224×224 para CXR (estándar ImageNet), 256×256 para CT (ya preprocesado)
