from typing import Dict

import cv2
import numpy as np


def postprocess_binary_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    keep_largest_component: bool = False,
    close_kernel_size: int = 0,
) -> np.ndarray:
    processed = mask.astype(np.uint8)

    if close_kernel_size > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
    if num_labels <= 1:
        return processed.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.ones(num_labels - 1, dtype=bool)
    if min_component_area > 0:
        keep &= areas >= min_component_area
    if keep_largest_component and keep.any():
        largest_idx = int(np.argmax(areas))
        keep[:] = False
        keep[largest_idx] = areas[largest_idx] >= min_component_area

    output = np.zeros_like(processed, dtype=np.uint8)
    for component_idx, should_keep in enumerate(keep, start=1):
        if should_keep:
            output[labels == component_idx] = 1
    return output.astype(bool)


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> Dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(axis=(1, 2))
    pred_sum = prediction.sum(axis=(1, 2))
    target_sum = target.sum(axis=(1, 2))
    union = pred_sum + target_sum - intersection

    dice = np.mean((2 * intersection + smooth) / (pred_sum + target_sum + smooth))
    iou = np.mean((intersection + smooth) / (union + smooth))
    pixel_accuracy = np.mean(prediction == target)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
    }
