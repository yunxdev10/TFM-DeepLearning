# Propuesta de mejora: trabajos relacionados y gaps identificados

Fecha: 2026-05-30

## Veredicto

El apartado actual de `Trabajos relacionados y gaps identificados` es util y esta bien enfocado, pero puede mejorarse. No sobra la idea general; lo que sobra es un poco de densidad y repeticion. La seccion actual intenta demostrar demasiadas cosas a la vez: que hay trabajos de CXR, CT, segmentacion, XAI, desbalanceo, trazabilidad, gaps y posicionamiento. Eso es defendible, pero puede quedar como una matriz de checklist.

La mejora recomendada es hacer la seccion mas narrativa: primero explicar los grupos de trabajos, despues mostrar una tabla compacta y finalmente cerrar con gaps claros. El tribunal no necesita ver una tabla enorme; necesita entender rapidamente que tu TFM no inventa una arquitectura nueva, sino que integra de forma reproducible varias dimensiones que normalmente aparecen separadas.

## Que mantendria

- Mantener la idea de que la literatura esta fragmentada por modalidad y tarea.
- Mantener la comparacion CXR frente a CT.
- Mantener la distincion entre clasificacion, segmentacion, XAI, desbalanceo y trazabilidad.
- Mantener la tabla comparativa, pero hacerla mas compacta.
- Mantener la frase de posicionamiento: el TFM es integrador, no una propuesta de arquitectura nueva.
- Mantener la cautela: no decir que tu TFM supera a todos los trabajos previos, sino que combina dimensiones bajo un protocolo comun.

## Que recortaria o cambiaria

### 1. Reducir la primera tabla

La tabla actual tiene demasiadas columnas:

- CXR
- CT
- Seg.
- XAI
- Desb.
- CXR-CT
- Traz.

Es visualmente densa y puede parecer demasiado mecanica. Recomiendo una tabla con 5 columnas:

| Trabajo | Modalidad | Tarea principal | Limitacion respecto a este TFM | Aporte de este TFM |
|---|---|---|---|---|

Con eso basta para defender el gap.

### 2. Evitar demasiados simbolos

Los simbolos `✓`, `~` y `-` son utiles, pero si la tabla es grande se vuelve dificil de leer. Si se mantienen, usar solo una tabla pequena. Para la memoria final, una tabla narrativa suele quedar mas academica.

### 3. No meter demasiados trabajos

No necesitas citar 15 trabajos en la tabla. Mejor 8-10 bien elegidos:

- Un trabajo CXR clasico o fuerte.
- El dataset CXR.
- MosMedData.
- Un trabajo CT de severidad.
- Un trabajo CT de segmentacion.
- Un trabajo de Attention U-Net/segmentacion.
- Un trabajo de Grad-CAM/XAI.
- Un trabajo critico sobre shortcut learning.
- Una revision critica/metodologica.
- Tu TFM.

### 4. No exagerar "CXR-CT" como gap absoluto

Hay trabajos que usan CXR y CT en un mismo articulo. Por eso conviene formularlo asi:

> Aunque existen trabajos que consideran mas de una modalidad, con frecuencia se centran en una tarea concreta o no integran de forma conjunta clasificacion, segmentacion, desbalanceo, explicabilidad y trazabilidad experimental.

Esto es mas prudente.

### 5. Separar "trabajos relacionados" de "gaps"

Ahora aparecen muy unidos. Recomiendo:

- `2.7.1 Trabajos relacionados por lineas de investigacion`
- `2.7.2 Sintesis comparativa`
- `2.7.3 Gaps metodologicos`
- `2.7.4 Posicionamiento del TFM`

## Version recomendada para sustituir el apartado

Puedes adaptar este texto directamente a LaTeX.

```latex
\subsection{Trabajos relacionados y gaps identificados}

La literatura sobre aprendizaje profundo aplicado a imagen toracica en COVID-19 es amplia, pero se encuentra distribuida en varias lineas de trabajo que no siempre se estudian de forma integrada. Una parte importante de los estudios se ha centrado en la clasificacion de radiografias de torax, especialmente mediante arquitecturas CNN preentrenadas y aprendizaje por transferencia. Otros trabajos se han orientado a la clasificacion o estimacion de severidad en tomografia computarizada, donde la informacion volumetrica y las etiquetas radiologicas plantean un escenario distinto al de la CXR. De forma paralela, existen estudios especificos sobre segmentacion de pulmon o lesion, explicabilidad visual y problemas metodologicos como el desbalanceo de clases, la fuga de datos o el aprendizaje espurio.

En CXR, trabajos como los basados en COVID-Net o en modelos preentrenados han mostrado el potencial de las CNN para detectar patrones compatibles con COVID-19 o neumonia en radiografias \cite{chowdhury2020canai,rahman2021covidradiography}. Asimismo, el uso de arquitecturas como ResNet, DenseNet o EfficientNet se ha consolidado como una estrategia frecuente por su disponibilidad, coste razonable y capacidad de reutilizar caracteristicas aprendidas en grandes datasets \cite{he2016resnet,huang2017densenet,tan2019efficientnet}. No obstante, muchos de estos trabajos se centran en una unica modalidad y reportan principalmente metricas de rendimiento global, lo que puede ocultar problemas de desbalanceo, confusiones entre clases o dependencia de patrones no clinicos.

En CT, datasets como MosMedData han permitido estudiar la severidad radiologica asociada a COVID-19 mediante etiquetas CT-0 a CT-4 y un subconjunto de mascaras de infeccion \cite{morozov2020mosmeddata}. Esta modalidad ofrece mayor detalle anatomico que la CXR, pero tambien introduce dificultades adicionales: los estudios son volumenes 3D, las etiquetas pueden describir grados de afectacion y no diagnosticos equivalentes a las clases CXR, y el procesamiento mediante slices 2D puede perder informacion entre cortes. Por ello, los resultados CT no deben compararse con CXR como si ambas tareas fueran clinicamente identicas.

La segmentacion constituye otra linea relevante. U-Net y sus variantes se han convertido en referencias para segmentacion biomedica, mientras que Attention U-Net introduce mecanismos de atencion que pueden ayudar a filtrar regiones poco relevantes en las conexiones de salto \cite{ronneberger2015unet,oktay2018attentionunet}. En COVID-19, varios estudios han aplicado segmentacion para delimitar pulmones o lesiones en CT, habitualmente con el objetivo de cuantificar la extension de la afectacion. Sin embargo, es importante diferenciar la segmentacion pulmonar, que delimita una region anatomica, de la segmentacion de lesion o infeccion, que aproxima una region patologica. Esta distincion es central en este TFM porque las mascaras CXR disponibles son pulmonares, mientras que las mascaras CT corresponden a lesion o infeccion.

Tambien se han propuesto tecnicas de explicabilidad visual, como Grad-CAM, para inspeccionar que regiones influyen en la prediccion de una CNN \cite{selvaraju2017gradcam}. En imagen medica, estas tecnicas son especialmente relevantes porque un modelo con alto rendimiento puede basar sus decisiones en atajos visuales, marcas del dataset o diferencias de adquisicion en lugar de senales clinicamente robustas \cite{degrave2021shortcuts,roberts2021commonpitfalls}. Sin embargo, los mapas de saliencia no equivalen a segmentaciones ni constituyen una prueba causal. Por ello, resulta necesario interpretarlos con cautela y, cuando existen mascaras, compararlos de forma cuantitativa con regiones anatomicas o patologicas disponibles.

La Tabla~\ref{tab:trabajos_relacionados_resumen} resume el posicionamiento de este TFM frente a trabajos representativos. La comparacion no pretende afirmar que los trabajos previos sean incompletos, sino mostrar que suelen abordar una parte del problema: una modalidad, una tarea, una arquitectura o una tecnica concreta. La aportacion de este TFM se situa en la integracion de varias dimensiones bajo un mismo flujo experimental.

\begin{table}[htbp]
\centering
\caption{Resumen comparativo de trabajos relacionados y posicionamiento del TFM.}
\label{tab:trabajos_relacionados_resumen}
\begin{tabular}{p{3.1cm} p{1.5cm} p{3.0cm} p{4.2cm} p{4.2cm}}
\hline
\textbf{Trabajo} & \textbf{Modalidad} & \textbf{Tarea principal} & \textbf{Limitacion respecto a este TFM} & \textbf{Aporte diferencial del TFM} \\
\hline
Chowdhury/Rahman et al. & CXR & Clasificacion y dataset CXR & Se centra en CXR y no estudia CT bajo el mismo protocolo. & Usa CXR dentro de una comparacion multimodal con CT, segmentacion y XAI. \\
Morozov et al. & CT & Dataset de severidad CT & Es una fuente de datos, no una comparacion completa de modelos. & Construye un pipeline CT con clasificacion, segmentacion y analisis frente a CXR. \\
Trabajos con ResNet/DenseNet/EfficientNet & CXR/CT & Clasificacion con transfer learning & Suelen centrarse en rendimiento de clasificacion. & Compara arquitecturas junto con estrategias de balanceo y metricas por clase. \\
Saood y Hatem & CT & Segmentacion de infeccion & Segmentacion CT sin integracion con CXR, clasificacion y XAI. & Integra segmentacion CT en un marco mas amplio con clasificacion y Grad-CAM. \\
Oktay et al. / Attention U-Net & Medica & Segmentacion con atencion & Arquitectura/metodo general, no comparacion CXR--CT. & Evalua U-Net y Attention U-Net en CXR pulmonar y CT lesion. \\
Selvaraju et al. / Grad-CAM & General & Explicabilidad visual & Metodo general, no especifico de COVID ni de mascaras medicas. & Aplica Grad-CAM y compara saliencia con mascaras disponibles. \\
DeGrave et al. / Roberts et al. & CXR/CT & Analisis critico y sesgos & No proponen el pipeline experimental de este TFM. & Motivan la cautela metodologica, el control de fuga y la evaluacion mas alla de accuracy. \\
\textbf{Este TFM} & CXR y CT & Clasificacion, segmentacion, XAI, desbalanceo y calibracion & No busca validacion clinica externa ni arquitectura nueva. & Integra multiples componentes en un protocolo reproducible y comparativo. \\
\hline
\end{tabular}
\end{table}

A partir de esta revision se identifican cuatro gaps principales. El primero es la comparacion multimodal controlada: CXR y CT suelen estudiarse por separado, y cuando aparecen juntas no siempre se diferencian sus etiquetas, significado clinico y limitaciones. El segundo es la integracion entre clasificacion, segmentacion y explicabilidad: muchos estudios reportan accuracy o mapas visuales, pero no relacionan las saliencias con mascaras anatomicas o patologicas disponibles. El tercero es el manejo explicito del desbalanceo: en datasets medicos, la accuracy puede ocultar bajo rendimiento en clases minoritarias, por lo que son necesarias metricas macro, analisis por clase y estrategias de balanceo. El cuarto gap es la trazabilidad experimental: las revisiones criticas sobre IA para COVID-19 destacan problemas de fuga de datos, sesgo de procedencia, validacion insuficiente y documentacion incompleta.

Este TFM se posiciona precisamente en esa interseccion. Su objetivo no es proponer una arquitectura completamente nueva, sino construir y evaluar un marco experimental que combine clasificacion multiclase, segmentacion, explicabilidad Grad-CAM, desbalanceo y calibracion probabilistica en dos modalidades de imagen toracica. Esta integracion permite interpretar los resultados desde varias perspectivas: rendimiento predictivo, equilibrio entre clases, calidad de segmentacion, plausibilidad espacial de las explicaciones y confianza del modelo. Por tanto, la contribucion principal es metodologica y comparativa, con una lectura prudente de las diferencias entre CXR y CT y de las limitaciones propias de los datasets disponibles.
```

## Notas importantes para adaptar el texto

- Revisa que las claves `\cite{...}` coincidan con tu `.bib` real. Algunas claves ya existen en `docs/bibliografia_base_tfm.bib`, pero otras pueden tener nombres distintos en tu fuente principal.
- Si no incluyes calibracion en el documento final, elimina la palabra `calibracion` del ultimo parrafo y de la fila `Este TFM`.
- Si tu tabla queda muy ancha en LaTeX, usa `tabularx`, `resizebox{\textwidth}{!}{...}` o convierte la tabla en una lista con parrafos.
- No pongas que tu TFM usa SHAP o LIME. En esta version solo aparece Grad-CAM, que es lo correcto.
- No pongas que tu TFM usa 3D U-Net. Puede citarse como contexto, pero tu metodologia es 2D y una variante 2.5D.

## Version aun mas corta si necesitas reducir paginas

Si el capitulo queda largo, puedes dejar solo:

- 2 parrafos de trabajos CXR/CT.
- 1 parrafo de segmentacion/XAI.
- 1 tabla compacta.
- 1 parrafo final de gaps.

La parte que mas valor aporta es el cierre:

> La contribucion principal del TFM no reside en proponer una arquitectura nueva, sino en integrar bajo un mismo protocolo comparativo la clasificacion, el manejo del desbalanceo, la segmentacion, la explicabilidad mediante Grad-CAM y la calibracion probabilistica en CXR y CT. Esta integracion permite analizar no solo si el modelo acierta, sino tambien como se comporta ante clases minoritarias, si sus predicciones se apoyan en regiones espacialmente plausibles y hasta que punto su confianza es fiable.
