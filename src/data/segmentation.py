from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data.ct_preprocessing import window_hu


CXR_CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def build_cxr_segmentation_dataframe(cxr_dir: Path) -> pd.DataFrame:
    records = []
    for label in CXR_CLASSES:
        image_dir = cxr_dir / label / "images"
        mask_dir = cxr_dir / label / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            continue

        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                continue
            records.append(
                {
                    "dataset": "cxr",
                    "label": label,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "sample_id": image_path.stem,
                    "study_id": image_path.stem,
                }
            )

    return pd.DataFrame(records)


def build_ct_segmentation_dataframe(
    ct_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (256, 256),
    slice_range: Tuple[float, float] = (0.2, 0.8),
    positive_mask_only: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    metadata_path = output_dir / "ct_segmentation_metadata.csv"
    if metadata_path.exists() and not overwrite:
        return pd.read_csv(metadata_path)

    image_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    mask_paths = sorted((ct_dir / "masks").glob("*_mask.nii*"))
    for mask_path in mask_paths:
        study_id = mask_path.name.replace("_mask.nii.gz", "").replace("_mask.nii", "")
        study_path = _find_ct_study_path(ct_dir, study_id)
        if study_path is None:
            continue

        volume = nib.load(str(study_path)).get_fdata()
        mask_volume = nib.load(str(mask_path)).get_fdata()
        if volume.shape != mask_volume.shape:
            raise ValueError(f"Shape mismatch for {study_id}: {volume.shape} vs {mask_volume.shape}")

        image_volume = window_hu(volume)
        num_slices = image_volume.shape[-1]
        start_idx = int(num_slices * slice_range[0])
        end_idx = int(num_slices * slice_range[1])

        for z in range(start_idx, end_idx):
            mask_slice = (mask_volume[:, :, z] > 0).astype(np.uint8)
            has_mask = bool(mask_slice.sum() > 0)
            if positive_mask_only and not has_mask:
                continue

            image_slice = np.rot90(image_volume[:, :, z])
            mask_slice = np.rot90(mask_slice)

            image_slice = cv2.resize(image_slice, target_size, interpolation=cv2.INTER_AREA)
            mask_slice = cv2.resize(mask_slice, target_size, interpolation=cv2.INTER_NEAREST)

            image_path = image_out_dir / f"{study_id}_slice_{z:03d}.png"
            mask_png_path = mask_out_dir / f"{study_id}_slice_{z:03d}_mask.png"
            cv2.imwrite(str(image_path), image_slice)
            cv2.imwrite(str(mask_png_path), (mask_slice * 255).astype(np.uint8))

            records.append(
                {
                    "dataset": "ct",
                    "label": "infection",
                    "image_path": str(image_path),
                    "mask_path": str(mask_png_path),
                    "sample_id": f"{study_id}_slice_{z:03d}",
                    "study_id": study_id,
                    "slice_index": z,
                    "has_mask": has_mask,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    return df


def split_segmentation_dataframe(
    df: pd.DataFrame,
    random_seed: int = 42,
    group_col: Optional[str] = None,
    stratify_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_col:
        group_df = df[[group_col, stratify_col]].drop_duplicates() if stratify_col else df[[group_col]].drop_duplicates()
        stratify = group_df[stratify_col] if stratify_col and group_df[stratify_col].nunique() > 1 else None
        train_groups, temp_groups = train_test_split(
            group_df,
            test_size=0.30,
            random_state=random_seed,
            stratify=stratify,
        )
        temp_stratify = temp_groups[stratify_col] if stratify_col and temp_groups[stratify_col].nunique() > 1 else None
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=0.50,
            random_state=random_seed,
            stratify=temp_stratify,
        )
        return (
            df[df[group_col].isin(train_groups[group_col])].reset_index(drop=True),
            df[df[group_col].isin(val_groups[group_col])].reset_index(drop=True),
            df[df[group_col].isin(test_groups[group_col])].reset_index(drop=True),
        )

    stratify = df[stratify_col] if stratify_col and df[stratify_col].nunique() > 1 else None
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_seed, stratify=stratify)
    temp_stratify = temp_df[stratify_col] if stratify_col and temp_df[stratify_col].nunique() > 1 else None
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed, stratify=temp_stratify)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class SegmentationPairTransform:
    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int,
        is_train: bool = True,
        rotation_degrees: float = 10.0,
        hflip_p: float = 0.5,
        train_crop_size: Optional[Tuple[int, int]] = None,
        train_crop_prob: float = 1.0,
        positive_crop_prob: float = 0.0,
    ):
        self.image_size = image_size
        self.in_channels = in_channels
        self.is_train = is_train
        self.rotation_degrees = rotation_degrees
        self.hflip_p = hflip_p
        self.train_crop_size = train_crop_size
        self.train_crop_prob = train_crop_prob
        self.positive_crop_prob = positive_crop_prob

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        if self.is_train:
            if torch.rand(1).item() < self.hflip_p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = float((torch.rand(1).item() * 2 - 1) * self.rotation_degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            did_crop = False
            if self.train_crop_size is not None and torch.rand(1).item() < self.train_crop_prob:
                image, mask = self._crop_pair(image, mask)
                did_crop = True
            elif self.train_crop_size is not None and not did_crop:
                image = TF.resize(image, self.train_crop_size, interpolation=InterpolationMode.BILINEAR)
                mask = TF.resize(mask, self.train_crop_size, interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        if self.in_channels == 1 and image_tensor.shape[0] != 1:
            image_tensor = TF.rgb_to_grayscale(image_tensor)
        elif self.in_channels == 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor

    def _crop_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        crop_h, crop_w = self.train_crop_size
        width, height = image.size
        crop_h = min(crop_h, height)
        crop_w = min(crop_w, width)

        mask_array = np.asarray(mask) > 0
        use_positive = mask_array.any() and torch.rand(1).item() < self.positive_crop_prob
        if use_positive:
            positive_y, positive_x = np.where(mask_array)
            selected_idx = int(torch.randint(0, len(positive_y), (1,)).item())
            center_y = int(positive_y[selected_idx])
            center_x = int(positive_x[selected_idx])
        else:
            center_y = int(torch.randint(0, height, (1,)).item())
            center_x = int(torch.randint(0, width, (1,)).item())

        top = int(np.clip(center_y - crop_h // 2, 0, height - crop_h))
        left = int(np.clip(center_x - crop_w // 2, 0, width - crop_w))
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        return image, mask


class SegmentationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        in_channels: int = 1,
        ct_context_slices: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.in_channels = in_channels
        self.ct_context_slices = ct_context_slices

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        if self.ct_context_slices and self.in_channels == 3:
            image = self._load_ct_context_image(row)
        else:
            image_mode = "L" if self.in_channels == 1 else "RGB"
            image = Image.open(row["image_path"]).convert(image_mode)
        mask = Image.open(row["mask_path"]).convert("L")

        if self.transform:
            return self.transform(image, mask)

        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return image_tensor, mask_tensor

    def _load_ct_context_image(self, row: pd.Series) -> Image.Image:
        image_path = Path(row["image_path"])
        study_id = row.get("study_id")
        slice_index = row.get("slice_index")
        if pd.isna(study_id) or pd.isna(slice_index):
            return Image.open(image_path).convert("RGB")

        current = Image.open(image_path).convert("L")
        channels = []
        for offset in (-1, 0, 1):
            neighbor_path = image_path.with_name(f"{study_id}_slice_{int(slice_index) + offset:03d}.png")
            if neighbor_path.exists():
                channels.append(np.asarray(Image.open(neighbor_path).convert("L")))
            else:
                channels.append(np.asarray(current))

        return Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8), mode="RGB")


def _find_ct_study_path(ct_dir: Path, study_id: str) -> Optional[Path]:
    for class_dir in ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]:
        candidate = ct_dir / "studies" / class_dir / f"{study_id}.nii"
        if candidate.exists():
            return candidate
        candidate_gz = ct_dir / "studies" / class_dir / f"{study_id}.nii.gz"
        if candidate_gz.exists():
            return candidate_gz
    return None
