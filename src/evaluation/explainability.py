from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.classifiers import CovidClassifier


class GradCAM:
    """Minimal Grad-CAM implementation for the local classifier wrappers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = normalize_tensor_01(cam)
        return cam.detach(), logits.detach()


def normalize_tensor_01(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat_min = tensor.amin(dim=tuple(range(1, tensor.ndim)), keepdim=True)
    flat_max = tensor.amax(dim=tuple(range(1, tensor.ndim)), keepdim=True)
    return (tensor - flat_min) / (flat_max - flat_min + eps)


def resolve_gradcam_target_layer(model: CovidClassifier) -> torch.nn.Module:
    architecture = model.architecture_name
    if architecture == "resnet50":
        return model.model.layer4[-1]
    if architecture == "densenet121":
        return model.model.features.denseblock4
    if architecture == "efficientnet_b0":
        return model.model.features[-1]
    raise ValueError(f"No Grad-CAM target layer configured for architecture={architecture!r}.")


def load_classifier_model(
    model_path: Path,
    architecture: str,
    num_classes: int,
    in_channels: int,
    device: str,
) -> Tuple[CovidClassifier, Dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = CovidClassifier(
        architecture_name=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
    )
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device).eval(), checkpoint if isinstance(checkpoint, dict) else {}


def denormalize_image(tensor: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    image = tensor.detach().cpu().clone()
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    image = image * std_tensor + mean_tensor
    image = image.clamp(0, 1)
    array = image.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        return array[..., 0]
    return array


def load_binary_mask(mask_path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L").resize(image_size[::-1], Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def saliency_to_binary_mask(saliency: np.ndarray, quantile: float = 0.80) -> np.ndarray:
    saliency = np.asarray(saliency, dtype=np.float32)
    if float(saliency.max()) <= 0:
        return np.zeros_like(saliency, dtype=bool)
    threshold = float(np.quantile(saliency, quantile))
    return saliency >= threshold


def binary_iou(pred_mask: np.ndarray, target_mask: np.ndarray, smooth: float = 1.0) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def saliency_inside_mask_ratio(saliency: np.ndarray, target_mask: np.ndarray, eps: float = 1e-8) -> float:
    saliency = np.asarray(saliency, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    return float(saliency[target].sum() / (saliency.sum() + eps))


def saliency_peak_inside_mask(saliency: np.ndarray, target_mask: np.ndarray) -> bool:
    peak_index = np.unravel_index(int(np.asarray(saliency).argmax()), np.asarray(saliency).shape)
    return bool(np.asarray(target_mask, dtype=bool)[peak_index])
