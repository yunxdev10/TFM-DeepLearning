import os
from pathlib import Path
from dataclasses import dataclass

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    """
    Centralized configuration class for the Deep Learning TFM.
    Holds paths, hyperparameters, and experiment settings.
    """
    # Base paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Specific Dataset Paths
    CXR_DIR: Path = BASE_DIR / "data" / "kaggle_data" / "COVID-19_Radiography_Dataset"
    CT_DIR: Path = BASE_DIR / "data" / "MosMedData_Chest_Scan"
    
    # Training Hyperparameters
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = min(4, os.cpu_count() or 1)
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    EPOCHS: int = 20
    EARLY_STOPPING_PATIENCE: int = 5
    
    # Input Image Dimensions
    CXR_IMAGE_SIZE: tuple = (224, 224)
    CT_IMAGE_SIZE: tuple = (256, 256)

    # Segmentation Hyperparameters
    SEGMENTATION_BATCH_SIZE: int = 8
    SEGMENTATION_LEARNING_RATE: float = 1e-4
    SEGMENTATION_WEIGHT_DECAY: float = 1e-5
    SEGMENTATION_EPOCHS: int = 30
    SEGMENTATION_EARLY_STOPPING_PATIENCE: int = 6
    
    # Random Seed for Reproducibility
    RANDOM_SEED: int = 42

config = Config()

# Ensure directories exist
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
