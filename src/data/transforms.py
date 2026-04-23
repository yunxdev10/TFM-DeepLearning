from torchvision import transforms
from typing import Tuple

# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_cxr_transforms(image_size: Tuple[int, int] = (224, 224), is_train: bool = True) -> transforms.Compose:
    """
    Data augmentations and transforms for CXR datasets.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation and Test
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def get_ct_transforms(image_size: Tuple[int, int] = (256, 256), is_train: bool = True) -> transforms.Compose:
    """
    Data augmentations and transforms for CT datasets.
    Augmentations are more conservative here because anatomy structure is critical.
    Note: CT images typically loaded as 1 channel (grayscale). We use a grayscale normalization if 1 channel,
    but assuming models are modified to handle 1 channel. If replicating channels, use ImageNet stats.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            # Less aggressive rotation and affine transformations for CT
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            # Normalizing grayscale images (assuming values roughly in [0,1] after ToTensor)
            # Or use ImageNet if we expand to 3 channels. We assume 1 channel here with simple 0.5 normalize.
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
