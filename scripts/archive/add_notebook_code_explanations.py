#!/usr/bin/env python3
"""Add beginner-friendly code explanations to every project notebook.

The script is intentionally idempotent: it removes previously generated
explanation cells before inserting fresh ones. Existing human-written markdown,
code, outputs and metadata are preserved.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
AUTO_CODE = "<!-- AUTO_EXPLICACION_CODIGO -->"
AUTO_GUIDE = "<!-- AUTO_GUIA_NOTEBOOK -->"


NOTEBOOK_PURPOSES = {
    "00_eda.ipynb": "analisis exploratorio inicial de los datasets CXR y CT",
    "02_cxr_classification.ipynb": "entrenamiento y evaluacion de clasificadores para radiografias de torax",
    "03_ct_classification.ipynb": "entrenamiento y evaluacion de clasificadores para tomografias CT",
    "04_classification_results.ipynb": "analisis comparativo de resultados de clasificacion",
    "05_segmentation.ipynb": "entrenamiento base de segmentacion en CXR y CT",
    "05b_ct_tversky_variant.ipynb": "variante CT con perdida Tversky ponderada",
    "05c_ct_patch_training.ipynb": "segmentacion CT mediante entrenamiento por parches positivos",
    "05d_ct_mixed_context_training.ipynb": "segmentacion CT combinando contexto local y global",
    "05e_ct_postprocessing_analysis.ipynb": "analisis de umbrales y postprocesado en segmentacion CT",
    "05f_ct_25d_context_training.ipynb": "exploracion CT 2.5D usando slices vecinos como contexto",
    "05g_ct_mixed_context_ablation.ipynb": "ablacion de parametros de contexto mixto en CT",
    "05h_ct_mixed_ensemble.ipynb": "ensemble de modelos CT mediante promedio de probabilidades",
    "05i_ct_negative_slice_training.ipynb": "experimento CT incorporando slices negativos",
    "05j_ct_high_capacity_refinement.ipynb": "refinamiento CT con mayor capacidad y entrenamiento mas largo",
    "05k_ct_segmentation_visualization.ipynb": "visualizacion cualitativa de mascaras reales y predichas",
    "06_explainability.ipynb": "explicabilidad mediante Grad-CAM y comparacion con mascaras",
    "07_final_analysis.ipynb": "analisis final integrado de todos los resultados",
    "08_calibration_analysis.ipynb": "calibracion probabilistica y errores de alta confianza",
    "09_ct_study_level_analysis.ipynb": "evaluacion CT por estudio agregando predicciones de slices",
    "10_ct_informative_slice_selection.ipynb": "seleccion de slices informativos en CT y comparacion experimental",
}


LIBRARY_EXPLANATIONS = {
    "os": "funciones del sistema operativo, como variables de entorno y rutas",
    "sys": "configuracion del interprete de Python, incluida la ruta de importacion",
    "json": "leer y escribir datos en formato JSON",
    "math": "operaciones matematicas basicas",
    "random": "generacion de numeros aleatorios y control de aleatoriedad",
    "subprocess": "ejecutar scripts externos desde Python",
    "Path": "manejar rutas de archivos de forma segura y legible",
    "numpy": "calculo numerico con arrays",
    "np": "calculo numerico con arrays",
    "pandas": "trabajar con tablas de datos",
    "pd": "trabajar con tablas de datos",
    "matplotlib.pyplot": "crear graficas",
    "plt": "crear graficas",
    "seaborn": "crear graficas estadisticas con buen formato",
    "sns": "crear graficas estadisticas con buen formato",
    "torch": "entrenamiento y evaluacion de redes neuronales con PyTorch",
    "DataLoader": "crear lotes de datos para entrenar o evaluar modelos",
}


PARAMETER_EXPLANATIONS = {
    "dataset_name": "indica que dataset se usara en el experimento, por ejemplo CXR o CT",
    "architecture": "elige la arquitectura del modelo que se va a entrenar o evaluar",
    "run_mode": "marca el modo de ejecucion; en la memoria se documentan los experimentos completos",
    "image_size": "define el tamano al que se redimensionan las imagenes de entrada",
    "in_channels": "indica cuantos canales tiene cada imagen de entrada; CT suele usar 1 y 2.5D puede usar 3",
    "variant_name": "da un nombre unico a la variante para guardar y comparar sus resultados",
    "loss_name": "selecciona la funcion de perdida que guiara el aprendizaje",
    "epochs": "define el numero maximo de pasadas completas por el conjunto de entrenamiento",
    "batch_size": "define cuantas imagenes procesa el modelo antes de actualizar pesos",
    "learning_rate": "controla el tamano de los pasos que da el optimizador al aprender",
    "weight_decay": "aplica regularizacion para reducir sobreajuste",
    "patience": "define cuantas epocas esperar sin mejora antes de parar con early stopping",
    "threshold": "fija el umbral para convertir probabilidades en mascara binaria",
    "base_features": "controla la capacidad inicial del modelo de segmentacion",
    "patch_size": "define el tamano de los recortes usados en entrenamiento por parches",
    "positive_fraction": "controla la proporcion de parches que contienen lesion",
    "mixed_full_probability": "define con que frecuencia se usa imagen completa frente a parche local",
    "bce_weight": "pondera la parte BCE de la perdida combinada",
    "tversky_weight": "pondera la parte Tversky de la perdida combinada",
    "dice_weight": "pondera la parte Dice/Tversky dentro de la perdida combinada",
    "pos_weight": "aumenta el peso de pixeles positivos o lesionados durante la perdida BCE",
    "alpha": "controla la penalizacion de falsos positivos en Tversky",
    "beta": "controla la penalizacion de falsos negativos en Tversky",
    "tversky_alpha": "controla cuanto penaliza la perdida Tversky los falsos positivos",
    "tversky_beta": "controla cuanto penaliza la perdida Tversky los falsos negativos",
    "optimize_threshold": "activa la busqueda del mejor umbral usando validacion",
    "threshold_search_max": "limita el valor maximo del barrido de umbrales",
    "train_crop_size": "define el tamano del recorte usado durante entrenamiento",
    "train_crop_prob": "define la probabilidad de entrenar con recorte en vez de imagen completa",
    "positive_crop_prob": "define la probabilidad de que el recorte contenga region positiva o lesion",
    "early_stopping_patience": "define cuantas epocas sin mejora se toleran antes de detener el entrenamiento",
    "selection_split": "indica la particion usada para elegir hiperparametros sin tocar test",
    "selection_metric": "indica la metrica usada para elegir la mejor configuracion",
    "train_df": "entrega la tabla de entrenamiento que el modelo usara para aprender",
    "val_df": "entrega la tabla de validacion usada para elegir checkpoints o hiperparametros",
    "test_df": "entrega la tabla de test reservada para la evaluacion final",
    "group_col": "indica la columna usada para agrupar, normalmente el estudio o paciente",
    "target_size": "indica el tamano final de imagen que espera el modelo",
    "positive_mask_only": "decide si se usan solo slices con mascara positiva",
    "random_seed": "fija la semilla aleatoria para hacer el resultado mas reproducible",
    "device": "indica si el calculo se ejecuta en CPU, MPS o GPU disponible",
    "run_config": "pasa la configuracion completa del experimento",
    "result": "pasa el resumen de metricas producido por el experimento",
    "latest_result": "pasa el ultimo resultado disponible para guardarlo o mostrarlo",
    "sample_result": "pasa un resultado de ejemplo para inspeccionarlo",
    "saved_paths": "pasa las rutas donde se han guardado artefactos del experimento",
    "artifact_paths": "pasa las rutas de archivos generados por el pipeline",
    "output_dir": "indica la carpeta donde se guardaran figuras, tablas o metricas",
    "overwrite": "decide si se reemplazan archivos existentes",
    "ct_context_slices": "define cuantos slices vecinos se usan como contexto 2.5D",
    "train_transform": "pasa las transformaciones aplicadas a las imagenes de entrenamiento",
    "eval_transform": "pasa las transformaciones usadas en validacion o test",
    "transform": "pasa la transformacion que se aplicara a cada imagen",
    "dataset_cls": "indica la clase Dataset que construye ejemplos para PyTorch",
    "label_map": "pasa el diccionario que traduce etiquetas a indices numericos",
    "ct_dir": "indica la carpeta donde estan los datos CT",
    "cls_dir": "indica la carpeta donde estan los resultados de clasificacion",
    "image_dir": "indica la carpeta donde estan las imagenes",
    "mask_dir": "indica la carpeta donde estan las mascaras",
    "checkpoint_path": "indica el archivo del modelo entrenado que se va a cargar",
    "model_path": "indica la ruta del modelo entrenado",
    "predictions_path": "indica la ruta del archivo de predicciones",
    "summary_path": "indica la ruta del resumen de resultados",
    "figure_path": "indica la ruta donde se guardara una figura",
    "confusion_path": "indica donde se guardara la matriz de confusion",
    "model": "pasa el modelo que se va a entrenar, evaluar o visualizar",
    "checkpoint": "pasa el checkpoint con pesos guardados",
    "state_dict": "pasa el diccionario de pesos aprendidos por el modelo",
    "loader": "pasa el DataLoader que entrega lotes de datos",
    "dataset": "pasa el dataset que contiene imagenes y etiquetas",
    "predictions": "pasa las predicciones calculadas por el modelo",
    "pred_df": "pasa una tabla de predicciones para analizarla",
    "metrics": "pasa metricas ya calculadas",
    "summary": "pasa un resumen estructurado de resultados",
    "image": "pasa la imagen que se va a mostrar, transformar o evaluar",
    "mask": "pasa la mascara real o predicha asociada a la imagen",
    "masks": "pasa un conjunto de mascaras",
    "probs": "pasa probabilidades generadas por el modelo",
    "logits": "pasa salidas crudas del modelo antes de convertirlas a probabilidades",
    "outputs": "pasa las salidas del modelo",
    "targets": "pasa las etiquetas reales usadas para calcular metricas",
    "y_true_bin": "pasa etiquetas reales binarizadas para calcular metricas tipo ROC/AUC",
    "y_score": "pasa puntuaciones o probabilidades para calcular curvas y AUC",
    "confusion": "pasa la matriz de confusion",
    "examples": "pasa ejemplos seleccionados para visualizacion cualitativa",
    "axes": "pasa varios ejes de matplotlib donde se dibujaran graficas",
    "ax": "pasa un eje concreto de matplotlib donde se dibujara una grafica",
    "x": "indica la variable del eje horizontal en una grafica",
    "y": "indica la variable del eje vertical en una grafica",
    "hue": "indica la variable usada para colorear grupos en una grafica",
    "annot": "decide si se escriben los valores dentro de una grafica tipo heatmap",
    "fmt": "define el formato numerico mostrado en la grafica",
    "cmap": "elige el mapa de colores de la grafica",
    "xticklabels": "define las etiquetas visibles del eje X",
    "yticklabels": "define las etiquetas visibles del eje Y",
    "keep_largest_component": "decide si el postprocesado conserva solo la region conectada mas grande",
    "min_component_area": "define el area minima para no eliminar una componente conectada",
    "close_kernel_size": "define el tamano del kernel para cerrar pequenos huecos en postprocesado",
    "is_train": "indica si el dataset se construye en modo entrenamiento o evaluacion",
    "shuffle": "decide si los datos se barajan antes de formar lotes",
    "num_workers": "define cuantos procesos auxiliares cargan datos",
    "index": "indica el indice de fila o ejemplo que se quiere seleccionar",
    "columns": "indica que columnas se seleccionan o muestran",
    "values": "indica que valores se usaran en una tabla o grafica",
    "data": "pasa la tabla o estructura de datos usada por la funcion",
    "check": "pasa una comprobacion o resultado intermedio para verificar que algo existe o es correcto",
    "cwd": "indica la carpeta de trabajo desde la que se ejecuta un comando",
    "pred": "pasa una prediccion concreta del modelo",
    "history_df": "pasa la tabla con la evolucion del entrenamiento por epoca",
    "image_path": "indica la ruta de una imagen concreta",
    "artifact_name": "indica el nombre del artefacto que se quiere leer o guardar",
    "both_count": "pasa el numero de casos que cumplen dos condiciones a la vez",
    "current_masks": "pasa las mascaras disponibles para el ejemplo actual",
    "ensemble_probs": "pasa las probabilidades combinadas del ensemble",
    "has_next": "indica si existe un slice posterior disponible",
    "has_prev": "indica si existe un slice anterior disponible",
    "mask_path": "indica la ruta del archivo de mascara",
    "middle": "pasa el slice central cuando se usan varios slices como contexto",
    "next_count": "pasa el numero de slices posteriores disponibles",
    "next_path": "indica la ruta del slice posterior",
    "nii_files": "pasa la lista de archivos NIfTI encontrados",
    "out_channels": "indica cuantos canales de salida produce el modelo",
    "overrides": "pasa valores que sustituyen la configuracion por defecto",
    "path": "pasa una ruta de archivo o carpeta",
    "prefix": "pasa un prefijo usado para nombrar archivos de salida",
    "prev_count": "pasa el numero de slices anteriores disponibles",
    "prev_path": "indica la ruta del slice anterior",
    "prob_cols": "pasa las columnas que contienen probabilidades por clase",
    "processed": "pasa datos que ya han sido transformados o procesados",
    "sample_summary": "pasa un resumen de ejemplo para inspeccionarlo",
    "stratify_col": "indica la columna usada para mantener proporciones de clase al dividir los datos",
    "total": "pasa el numero total de elementos contabilizados",
    "volume": "pasa un volumen CT completo o una estructura volumetrica",
}


def text_source(source: object) -> str:
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return ""


def markdown_cell(lines: list[str], tag: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {"tags": [tag]},
        "source": [f"{line}\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def notebook_guide(nb_name: str) -> dict:
    purpose = NOTEBOOK_PURPOSES.get(nb_name, "una fase del flujo experimental del TFM")
    lines = [
        AUTO_GUIDE,
        "",
        "## Como leer este notebook",
        "",
        f"Este notebook forma parte del flujo del TFM y se centra en {purpose}.",
        "",
        "Antes de cada celda de codigo encontraras una guia con cuatro ideas:",
        "",
        "- **Que hace:** resume la funcion practica de la celda.",
        "- **Por que lo hacemos:** conecta el codigo con la metodologia del TFM.",
        "- **Resultado esperado:** indica que deberias ver al ejecutar la celda si todo va bien.",
        "- **Explicacion linea a linea:** traduce cada instruccion de Python a lenguaje sencillo.",
        "",
        "La idea es que puedas defender no solo el resultado, sino tambien el camino seguido para obtenerlo.",
    ]
    return markdown_cell(lines, "auto_guia_notebook")


def cell_kind(code: str) -> str:
    lower = code.lower()
    if re.search(r"(^|\n)\s*(import|from)\s+", code):
        return "setup_imports"
    if "train_and_evaluate" in code or ("train_" in lower and "segmentation" in lower):
        return "training"
    if "subprocess.run" in code:
        return "external_script"
    if "grad" in lower or "saliency" in lower or "explain" in lower:
        return "explainability"
    if "calibration" in lower or "ece" in lower or "brier" in lower:
        return "calibration"
    if "ensemble" in lower or "weight_grid" in lower:
        return "ensemble"
    if "confusion" in lower or "classification_report" in lower or "auc" in lower:
        return "classification_metrics"
    if "dice" in lower or "iou" in lower or "threshold" in lower or "mask" in lower:
        return "segmentation_metrics"
    if "plt." in code or "sns." in code or "figure" in lower:
        return "plotting"
    if "read_csv" in code or "dataframe" in lower or "metadata" in lower:
        return "data_loading"
    if "project_root" in lower or "results_dir" in lower:
        return "paths_config"
    if "display" in code or ".head(" in code or ".describe(" in code or ".value_counts(" in code:
        return "inspection"
    return "general"


def purpose_for(kind: str) -> str:
    return {
        "setup_imports": "prepara el entorno de trabajo: carga librerias, rutas del proyecto y configuracion visual.",
        "paths_config": "define rutas y variables de configuracion para que el resto del notebook sepa donde leer y guardar archivos.",
        "data_loading": "carga tablas, metadatos o resultados ya generados para poder analizarlos en el notebook.",
        "inspection": "muestra informacion intermedia para comprobar que los datos tienen la forma esperada.",
        "training": "lanza un entrenamiento o una evaluacion completa de un modelo.",
        "external_script": "ejecuta un script externo del proyecto desde el notebook para reutilizar el pipeline ya implementado.",
        "classification_metrics": "calcula o muestra metricas de clasificacion para comparar modelos y errores por clase.",
        "segmentation_metrics": "calcula o muestra metricas de segmentacion, especialmente Dice, IoU, umbral y tamano de mascaras.",
        "plotting": "genera graficas para interpretar visualmente resultados, distribuciones o comparaciones.",
        "explainability": "genera o analiza explicaciones Grad-CAM para estudiar donde mira el clasificador.",
        "calibration": "analiza si las probabilidades del modelo estan bien calibradas y si hay errores de alta confianza.",
        "ensemble": "combina predicciones de varios modelos para probar si el promedio mejora la estabilidad o el rendimiento.",
        "general": "realiza una operacion auxiliar necesaria para continuar el flujo del notebook.",
    }[kind]


def why_for(kind: str, nb_name: str) -> str:
    purpose = NOTEBOOK_PURPOSES.get(nb_name, "esta fase del TFM")
    return {
        "setup_imports": f"Sin estas librerias y rutas, Python no podria encontrar las funciones del proyecto ni crear tablas/graficas para {purpose}.",
        "paths_config": "Centralizar rutas evita errores al mover el proyecto y hace que los resultados sean reproducibles.",
        "data_loading": "Antes de entrenar o interpretar resultados hay que transformar archivos guardados en tablas que Python pueda manipular.",
        "inspection": "Estas comprobaciones ayudan a detectar problemas temprano: clases mal cargadas, columnas ausentes, particiones incorrectas o resultados incompletos.",
        "training": "El TFM se basa en experimentos reales; esta celda produce artefactos medibles como checkpoints, metricas y resumenes.",
        "external_script": "Usar scripts permite que el experimento sea repetible fuera del notebook y reduce codigo duplicado.",
        "classification_metrics": "La clasificacion medica no puede evaluarse solo con accuracy; necesitamos ver sensibilidad, F1, AUC y errores por clase.",
        "segmentation_metrics": "En segmentacion importa el solapamiento entre mascara predicha y real; por eso se priorizan Dice e IoU.",
        "plotting": "Las graficas permiten ver patrones que en tablas son dificiles de detectar, como desbalanceo, mejoras relativas o errores repetidos.",
        "explainability": "Grad-CAM ayuda a revisar si el modelo se apoya en regiones razonables, aunque no equivale a una prueba clinica.",
        "calibration": "Dos modelos pueden acertar parecido pero tener niveles de confianza muy distintos; la calibracion mide esa fiabilidad probabilistica.",
        "ensemble": "El ensemble prueba si combinar modelos entrenados con estrategias distintas reduce errores individuales.",
        "general": "Esta operacion conecta pasos anteriores con pasos posteriores del notebook.",
    }[kind]


def expected_for(kind: str) -> str:
    return {
        "setup_imports": "La celda debe ejecutarse sin errores. Si falla, normalmente falta una libreria o la ruta del proyecto no esta bien configurada.",
        "paths_config": "Se crean variables de ruta o configuracion. Normalmente no aparece una grafica, pero las siguientes celdas dependen de estas variables.",
        "data_loading": "Debe aparecer una tabla, un resumen o una variable cargada. Si el archivo no existe, revisa que los notebooks anteriores se hayan ejecutado.",
        "inspection": "Deberias ver tablas pequenas, conteos, nombres de columnas o ejemplos que confirmen que los datos son coherentes.",
        "training": "El entrenamiento puede tardar. Al final deberian guardarse metricas, modelos y resumenes en la carpeta de resultados.",
        "external_script": "Deberias ver mensajes del script indicando que se han generado resultados o que se han reutilizado artefactos existentes.",
        "classification_metrics": "Deberias obtener metricas numericas y, a veces, matrices o tablas para comparar modelos.",
        "segmentation_metrics": "Deberias obtener valores de Dice, IoU, pixel accuracy, umbrales o tablas por slice/modelo.",
        "plotting": "Deberia aparecer una figura o guardarse una imagen en disco. Si no aparece nada, revisa rutas y datos de entrada.",
        "explainability": "Deberian generarse mapas Grad-CAM y metricas de solapamiento con mascaras cuando existan.",
        "calibration": "Deberian aparecer tablas/graficas con ECE, Brier score, NLL o errores de alta confianza.",
        "ensemble": "Deberia aparecer una combinacion seleccionada por validacion y una evaluacion final sobre test.",
        "general": "La celda debe dejar preparado algun objeto, tabla, figura o archivo usado mas adelante.",
    }[kind]


def explain_import(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("import "):
        pieces = [piece.strip() for piece in stripped[len("import ") :].split(",")]
        explanations = []
        for piece in pieces:
            name, _, alias = piece.partition(" as ")
            key = alias.strip() or name.strip()
            description = LIBRARY_EXPLANATIONS.get(
                key, LIBRARY_EXPLANATIONS.get(name.split(".")[0].strip(), "funciones externas que se usan despues")
            )
            if alias:
                explanations.append(f"importa `{name.strip()}` con el alias `{alias.strip()}` para usar {description}.")
            else:
                explanations.append(f"importa `{name.strip()}` para usar {description}.")
        return " ".join(explanations)

    match = re.match(r"from\s+(.+?)\s+import\s+(.+)", stripped)
    if match:
        module, names = match.groups()
        return f"importa `{names}` desde `{module}` para poder usar esas funciones, clases o constantes sin escribir el modulo completo cada vez."
    return "carga herramientas externas que se usaran en las siguientes celdas."


def explain_keyword_argument(line: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
    if not match:
        return None
    name = match.group(1)
    if not line.startswith((" ", "\t")):
        return None
    detail = PARAMETER_EXPLANATIONS.get(name)
    if detail:
        return f"pasa el parametro `{name}` a la funcion; {detail}."
    return f"pasa el parametro `{name}` a la funcion que se esta llamando."


def explain_assignment(line: str) -> str:
    left = line.strip().split("=", 1)[0].strip()
    if left.isupper():
        return f"define la constante `{left}`; es un valor de configuracion que el notebook reutiliza mas adelante."
    return f"crea o actualiza la variable `{left}` con el resultado de la expresion de la derecha."


def compact_code(line: str, limit: int = 150) -> str:
    stripped = line.rstrip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def explain_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return "es un comentario: no se ejecuta, sirve para que la persona que lee entienda la intencion del codigo."
    if stripped.startswith(("import ", "from ")):
        return explain_import(line)
    if stripped.startswith("PROJECT_ROOT"):
        return "calcula la carpeta raiz del proyecto para construir rutas absolutas de forma consistente."
    if "sys.path" in stripped and ("append" in stripped or "insert" in stripped):
        return "anade la carpeta del proyecto a la ruta de Python para poder importar modulos propios como `src.config` o `src.training`."
    if "os.environ" in stripped:
        return "define una variable de entorno; normalmente se usa para configurar caches o comportamiento de librerias antes de ejecutarlas."
    if stripped.startswith("config."):
        return "ajusta una opcion global de configuracion del proyecto para esta ejecucion."
    if stripped.startswith("print("):
        return "muestra un mensaje o valor en pantalla para comprobar el estado del proceso."
    if stripped.startswith("display("):
        return "muestra una tabla, figura u objeto de forma mas legible dentro del notebook."
    if stripped.startswith("for "):
        return "inicia un bucle: Python repetira el bloque indentado para cada elemento de una coleccion."
    if stripped.startswith("if "):
        return "comprueba una condicion; si se cumple, se ejecuta el bloque indentado que viene debajo."
    if stripped.startswith("elif "):
        return "anade una condicion alternativa cuando el `if` anterior no se ha cumplido."
    if stripped.startswith("else"):
        return "define que hacer cuando ninguna de las condiciones anteriores se cumple."
    if stripped.startswith("with "):
        return "abre un contexto controlado; Python se encarga de cerrar o limpiar el recurso al terminar el bloque."
    if stripped.startswith("def "):
        name = stripped.split("def ", 1)[1].split("(", 1)[0]
        return f"define la funcion `{name}` para reutilizar ese bloque de instrucciones varias veces."
    if stripped.startswith("return "):
        return "devuelve un resultado desde una funcion a la parte del codigo que la llamo."
    if stripped.startswith("assert "):
        return "verifica una condicion obligatoria; si no se cumple, Python detiene la ejecucion con un error."
    if stripped.startswith("raise "):
        return "lanza un error de forma explicita para evitar continuar con datos o configuracion incorrecta."

    keyword_argument = explain_keyword_argument(line)
    if keyword_argument:
        return keyword_argument

    special_cases = [
        ("pd.read_csv", "lee un archivo CSV y lo convierte en un DataFrame, que es una tabla manipulable con pandas."),
        (".read_text", "lee el contenido de un archivo de texto desde disco."),
        ("json.loads", "convierte texto JSON en estructuras de Python como diccionarios o listas."),
        ("subprocess.run", "ejecuta un comando o script externo y permite reutilizar codigo preparado fuera del notebook."),
        ("train_and_evaluate", "llama a la funcion principal de entrenamiento/evaluacion para producir metricas y artefactos del experimento."),
        ("make_segmentation_run_config", "construye la configuracion completa del experimento de segmentacion."),
        ("make_run_config", "construye la configuracion completa del experimento de clasificacion."),
        ("DataLoader", "crea un cargador de datos que entrega imagenes en lotes al modelo durante entrenamiento o evaluacion."),
        ("torch.load", "carga desde disco pesos o informacion guardada de un modelo entrenado."),
        ("load_state_dict", "introduce en el modelo los pesos aprendidos durante el entrenamiento."),
        ("model.eval", "pone el modelo en modo evaluacion para desactivar comportamientos propios del entrenamiento."),
        ("torch.no_grad", "desactiva el calculo de gradientes para ahorrar memoria y acelerar la evaluacion."),
        ("plt.subplots", "crea una figura y uno o varios ejes donde se dibujaran las graficas."),
        ("sns.barplot", "dibuja un grafico de barras para comparar valores entre categorias o modelos."),
        ("sns.heatmap", "dibuja un mapa de calor, util para matrices de confusion o tablas de intensidades."),
        ("plt.tight_layout", "ajusta margenes para que titulos, etiquetas y graficas no se solapen."),
        ("plt.show", "muestra la figura generada en la salida del notebook."),
        (".savefig", "guarda la figura en un archivo de imagen para poder incluirla en el informe."),
        ("classification_report", "calcula precision, recall y F1 por clase para analizar el rendimiento mas alla de la accuracy."),
        ("confusion_matrix", "calcula la matriz de confusion para ver que clases se confunden entre si."),
        ("roc_auc", "calcula informacion de AUC para medir capacidad discriminativa."),
        ("np.arange", "crea una secuencia de numeros para probar varios valores, por ejemplo umbrales o pesos."),
        ("np.linspace", "crea una secuencia de numeros igualmente espaciados para explorar valores."),
        (".groupby", "agrupa filas de una tabla para calcular resumenes por clase, modelo, dataset u otra categoria."),
        (".merge", "combina dos tablas usando una o varias columnas en comun."),
        (".value_counts", "cuenta cuantas veces aparece cada valor, util para revisar distribuciones de clases."),
        (".head(", "muestra las primeras filas de una tabla para inspeccionar rapidamente su contenido."),
    ]
    for token, explanation in special_cases:
        if token in stripped:
            return explanation

    lower = stripped.lower()
    if "dice" in lower or "iou" in lower:
        return "trabaja con metricas de solapamiento entre mascara predicha y mascara real."
    if "threshold" in lower:
        return "define o evalua un umbral para convertir probabilidades en decisiones binarias."
    if "gradcam" in lower or "grad_cam" in lower:
        return "prepara o usa Grad-CAM para obtener un mapa de calor de las zonas influyentes en la prediccion."
    if "ece" in lower or "brier" in lower:
        return "trabaja con metricas de calibracion para evaluar si la confianza del modelo es fiable."
    if re.match(r"^[A-Za-z_][\w\.\[\]\"']*\s*=", stripped):
        return explain_assignment(line)
    if stripped in {"]", "}", ")", "],", "},", "),"}:
        return "cierra una estructura abierta antes, como una lista, un diccionario o una llamada a funcion."
    if stripped.startswith(("[", "{", "(")):
        return "empieza una estructura de datos o una llamada dividida en varias lineas para que sea mas legible."
    return "ejecuta esta instruccion como parte del flujo de la celda; normalmente transforma datos, calcula valores o prepara una salida."


def code_context(code: str) -> str:
    lower = code.lower()
    labels = []
    for token, label in [
        ("cxr", "CXR"),
        ("ct", "CT"),
        ("classification", "clasificacion"),
        ("segmentation", "segmentacion"),
        ("grad", "Grad-CAM"),
        ("calibration", "calibracion"),
        ("threshold", "umbral"),
        ("ensemble", "ensemble"),
    ]:
        if token in lower and label not in labels:
            labels.append(label)
    if labels:
        return "Temas principales detectados en la celda: " + ", ".join(labels) + "."
    return "La celda contiene operaciones auxiliares del pipeline."


def explanation_for_code_cell(code: str, code_index: int, nb_name: str) -> dict:
    kind = cell_kind(code)
    lines = [
        AUTO_CODE,
        "",
        f"### Guia de codigo: celda {code_index}",
        "",
        f"**Que hace:** {purpose_for(kind)}",
        "",
        f"**Por que lo hacemos:** {why_for(kind, nb_name)}",
        "",
        f"**Resultado esperado:** {expected_for(kind)}",
        "",
        f"**Contexto rapido:** {code_context(code)}",
        "",
        "**Explicacion linea a linea:**",
        "",
    ]
    explained_any_line = False
    for line_number, raw_line in enumerate(code.splitlines(), start=1):
        explanation = explain_line(raw_line)
        if explanation is None:
            continue
        explained_any_line = True
        lines.append(f"- Linea {line_number}: `{compact_code(raw_line)}`")
        lines.append(f"  - {explanation}")
    if not explained_any_line:
        lines.append("- Esta celda no contiene instrucciones ejecutables relevantes.")
    return markdown_cell(lines, "auto_explicacion_codigo")


def remove_generated_cells(cells: list[dict]) -> list[dict]:
    cleaned = []
    for cell in cells:
        source = text_source(cell.get("source", ""))
        if cell.get("cell_type") == "markdown" and (AUTO_CODE in source or AUTO_GUIDE in source):
            continue
        cleaned.append(cell)
    return cleaned


def insert_notebook_guide(cells: list[dict], nb_name: str) -> list[dict]:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            return cells[: index + 1] + [notebook_guide(nb_name)] + cells[index + 1 :]
    return [notebook_guide(nb_name)] + cells


def document_notebook(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = insert_notebook_guide(remove_generated_cells(data.get("cells", [])), path.name)

    documented = []
    code_index = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            code_index += 1
            documented.append(explanation_for_code_cell(text_source(cell.get("source", "")), code_index, path.name))
        documented.append(cell)

    data["cells"] = documented
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return code_index


def main() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {NOTEBOOK_DIR}")

    print("Notebooks actualizados:")
    for notebook in notebooks:
        count = document_notebook(notebook)
        print(f"- {notebook.name}: {count} celdas de codigo documentadas")


if __name__ == "__main__":
    main()
