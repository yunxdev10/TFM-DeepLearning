# Experimentos auxiliares de segmentacion CT

Estos notebooks documentan variantes probadas para mejorar la segmentacion de lesion/infeccion en CT.

No forman parte del recorrido principal de ejecucion, pero son importantes como evidencia experimental:

| Notebook | Idea experimental |
| --- | --- |
| `05b_ct_tversky_variant.ipynb` | Variante ligera con Tversky/BCE ponderada. |
| `05c_ct_patch_training.ipynb` | Entrenamiento por parches y muestreo positivo. |
| `05d_ct_mixed_context_training.ipynb` | Mezcla de contexto local y global. |
| `05e_ct_postprocessing_analysis.ipynb` | Barrido de umbral y postprocesado. |
| `05f_ct_25d_context_training.ipynb` | Exploracion 2.5D con slices vecinos. |
| `05g_ct_mixed_context_ablation.ipynb` | Ablacion del contexto mixto. |
| `05h_ct_mixed_ensemble.ipynb` | Ensemble de modelos CT. |
| `05i_ct_negative_slice_training.ipynb` | Inclusion de slices negativos. |
| `05j_ct_high_capacity_refinement.ipynb` | Refinamiento de capacidad y entrenamiento mas largo. |

Resultado principal para la memoria: el mejor modelo CT de segmentacion fue `ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation`, con Dice aproximado `0.5637` e IoU `0.4305`.
