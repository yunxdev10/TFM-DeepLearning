# Verificacion de fiabilidad de datasets frente a fuentes web

Fecha de verificacion: 2026-06-05

## Objetivo

Comprobar si los datasets usados en el TFM concuerdan con las fuentes publicas disponibles en web y detectar posibles matices metodologicos que deban documentarse en la memoria.

Datasets verificados:

- CXR: `COVID-19 Radiography Database`.
- CT: `MosMedData: Chest CT Scans with COVID-19 Related Findings`.

## Fuentes consultadas

### CXR

- Kaggle, COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Pagina de datos Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data
- Rahman et al., image enhancement con CXR: https://arxiv.org/abs/2012.02238
- Articulo que resume la distribucion del dataset: https://pmc.ncbi.nlm.nih.gov/articles/PMC11109273/

### CT

- MosMedData arXiv: https://arxiv.org/abs/2005.06465
- MosMedData Academic Torrents: https://academictorrents.com/details/f2175c4676e041ea65568bb70c2bcd15c7325fd2
- Articulo con resumen de distribucion MosMedData: https://www.mdpi.com/2306-5354/10/5/529

## Resultado ejecutivo

Los datasets locales concuerdan con las fuentes publicas en los puntos principales:

- La distribucion CXR local coincide exactamente con la declarada por la COVID-19 Radiography Database.
- La distribucion CT local por estudios coincide exactamente con MosMedData.
- Las mascaras CXR estan presentes para todas las imagenes locales.
- Las mascaras CT corresponden al subconjunto anotado de 50 estudios de MosMedData.
- Las diferencias de conteo entre estudios CT y slices PNG no son contradicciones, sino consecuencia del preprocesamiento local a 2D.

Conclusion: los datos son coherentes con las fuentes originales y se pueden describir como datasets publicos reconocidos. Aun asi, deben documentarse sus limitaciones: procedencia multicentro/multifuente en CXR, fuerte desbalance en CT, CT-4 con solo 2 estudios, y diferencia entre mascaras pulmonares CXR y mascaras patologicas CT.

## Verificacion CXR

### Lo que declara la fuente

La COVID-19 Radiography Database declara la siguiente distribucion:

| Clase | Numero declarado |
|---|---:|
| COVID | 3616 |
| Normal | 10192 |
| Lung Opacity | 6012 |
| Viral Pneumonia | 1345 |
| Total | 21165 |

La fuente tambien indica que existen mascaras pulmonares correspondientes y que las imagenes estan en formato PNG con resolucion 299 x 299.

### Lo que hay en el proyecto

Ruta local:

`data/kaggle_data/COVID-19_Radiography_Dataset`

| Clase | Imagenes locales | Mascaras locales | Coincide con fuente |
|---|---:|---:|---|
| COVID | 3616 | 3616 | Si |
| Lung_Opacity | 6012 | 6012 | Si |
| Normal | 10192 | 10192 | Si |
| Viral Pneumonia | 1345 | 1345 | Si |
| Total | 21165 | 21165 | Si |

Comprobaciones adicionales:

- Todas las imagenes CXR locales tienen tamano 299 x 299.
- Todas las mascaras CXR locales tienen tamano 256 x 256.
- La correspondencia por nombre entre imagen y mascara es completa en las cuatro clases.

### Interpretacion

El dataset CXR local es consistente con la distribucion oficial. Es correcto escribir que se usan 21165 radiografias de torax distribuidas en cuatro clases: COVID, Lung Opacity, Normal y Viral Pneumonia.

Matiz importante para la memoria:

- Las mascaras CXR son mascaras pulmonares, no mascaras de lesion COVID.
- Por tanto, en CXR la segmentacion y Grad-CAM frente a mascara evaluan plausibilidad anatomica dentro del campo pulmonar, no localizacion exacta de lesion patologica.
- El dataset CXR procede de multiples fuentes publicas; por eso es necesario discutir posibles sesgos de origen y shortcut learning.

## Verificacion CT

### Lo que declara la fuente

MosMedData declara:

| Clase original | Estudios declarados |
|---|---:|
| CT-0 | 254 |
| CT-1 | 684 |
| CT-2 | 125 |
| CT-3 | 45 |
| CT-4 | 2 |
| Total | 1110 |

Tambien declara:

- 1110 estudios/pacientes.
- 50 estudios con mascaras binarias de regiones de interes.
- Estudios en formato NIfTI comprimido.
- Categorias CT-0 a CT-4 basadas en severidad radiologica.
- Las mascaras anotan opacidades en vidrio deslustrado y consolidaciones.

### Lo que hay en el proyecto

Ruta local:

`data/MosMedData_Chest_Scan`

Conteo de estudios originales:

| Clase original | Estudios locales | Coincide con fuente |
|---|---:|---|
| CT-0 | 254 | Si |
| CT-1 | 684 | Si |
| CT-2 | 125 | Si |
| CT-3 | 45 | Si |
| CT-4 | 2 | Si |
| Total | 1110 | Si |

Conteo de slices 2D procesados para clasificacion:

| Clase usada en el TFM | Slices PNG | Estudios |
|---|---:|---:|
| CT-0 | 6642 | 254 |
| CT-1 | 16894 | 684 |
| CT-2 | 3096 | 125 |
| CT-3+ | 1149 | 47 |
| Total | 27781 | 1110 |

Conteo de segmentacion CT:

| Elemento | Conteo local |
|---|---:|
| Estudios con mascara | 50 |
| Slices imagen procesados | 704 |
| Slices mascara procesados | 704 |
| Correspondencia imagen-mascara normalizada | Completa |

### Interpretacion

El dataset CT local concuerda con MosMedData. Es correcto escribir que se usan 1110 estudios CT con clases originales CT-0, CT-1, CT-2, CT-3 y CT-4.

Tambien es correcto escribir que en el TFM se fusionan CT-3 y CT-4 como `CT-3+`, porque CT-4 solo contiene 2 estudios. Esta fusion no es una etiqueta original de MosMedData; es una decision metodologica del TFM para reducir la fragmentacion extrema de la clase mas severa.

Matiz importante:

- MosMedData esta definido a nivel de estudio CT volumetrico.
- En el TFM se transforma a slices 2D PNG para entrenar clasificadores 2D.
- Por eso el numero local de imagenes procesadas, 27781 slices, no debe confundirse con el numero original de estudios, 1110.
- La etiqueta de cada slice procede de la etiqueta del estudio completo.
- Para segmentacion, las mascaras solo existen en 50 estudios, y el preprocesamiento local genera 704 pares imagen-mascara positivos.

## Riesgos y limitaciones que conviene declarar

### CXR

- El dataset combina imagenes procedentes de distintas fuentes publicas.
- Esto puede introducir diferencias de adquisicion, marcas, resolucion, compresion o protocolos.
- Las mascaras son pulmonares, no patologicas.
- Un buen resultado CXR no implica automaticamente validacion clinica externa.

### CT

- La distribucion de severidad esta muy desbalanceada: CT-1 domina y CT-4 contiene solo 2 estudios.
- La conversion de volumenes 3D a slices 2D reduce el contexto volumetrico.
- La etiqueta de severidad se hereda del estudio completo a cada slice, aunque no todos los slices muestran la misma carga patologica.
- Solo hay 50 estudios con mascaras de infeccion, por lo que la segmentacion CT es una tarea limitada y dificil.

## Frases recomendadas para la memoria

### Dataset CXR

En este trabajo se utiliza la COVID-19 Radiography Database, un conjunto publico de radiografias de torax compuesto por 21165 imagenes PNG distribuidas en cuatro clases: 3616 COVID, 6012 Lung Opacity, 10192 Normal y 1345 Viral Pneumonia. La copia local empleada en el TFM reproduce exactamente esta distribucion e incluye mascaras pulmonares correspondientes para todas las imagenes. Estas mascaras se interpretan como delimitaciones anatomicas del campo pulmonar y no como anotaciones de lesion COVID.

### Dataset CT

Para la modalidad CT se utiliza MosMedData, formado por 1110 estudios de tomografia computarizada de torax clasificados en cinco niveles de severidad: CT-0, CT-1, CT-2, CT-3 y CT-4. La distribucion local coincide con la fuente original: 254, 684, 125, 45 y 2 estudios, respectivamente. Debido al numero extremadamente bajo de estudios CT-4, en este TFM se agrupan CT-3 y CT-4 en una clase `CT-3+` para obtener una formulacion experimental mas estable. Ademas, se emplea el subconjunto de 50 estudios con mascaras binarias de infeccion para la tarea de segmentacion CT.

### Sobre preprocesamiento CT

Aunque MosMedData esta compuesto por 1110 estudios volumetricos, el pipeline del TFM convierte los volumenes en slices 2D PNG para permitir el entrenamiento de modelos 2D con recursos locales. Por ello, el numero de imagenes procesadas es superior al numero de estudios originales. Esta conversion debe describirse como una decision metodologica y no como una diferencia respecto al dataset fuente.

## Dictamen final

La copia local de los datasets es coherente con las fuentes web consultadas. No se han detectado contradicciones relevantes en clases, conteos principales o disponibilidad de mascaras. Lo que si debe cuidarse en la redaccion es distinguir entre:

- estudios CT originales y slices 2D procesados,
- mascaras pulmonares CXR y mascaras de lesion/infeccion CT,
- clases originales de MosMedData y la fusion metodologica `CT-3+`,
- datos publicos reconocidos y validacion clinica real.

Esta verificacion refuerza la fiabilidad documental del TFM, siempre que las limitaciones anteriores se expliquen de forma transparente.
