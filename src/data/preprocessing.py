import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def get_cxr_dataframes(data_dir: Path, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans the Kaggle COVID-19 Radiography Dataset directory and generates 
    stratified train, validation, and test dataframes.
    
    Args:
        data_dir: Path to the COVID-19_Radiography_Dataset directory.
        random_seed: Seed for reproducible stratified splits.
        
    Returns:
        train_df, val_df, test_df
    """
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    
    data = []
    
    for cls in classes:
        img_dir = data_dir / cls / "images"
        if not img_dir.exists():
            print(f"Warning: Directory {img_dir} not found.")
            continue
            
        for img_name in os.listdir(img_dir):
            # Typical images are .png, ignore hidden files like .DS_Store
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                data.append({
                    'image_path': str(img_dir / img_name),
                    'label': cls
                })
                
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError(f"No images found in {data_dir}. Check the path and structure.")
        
    # Stratified Split: 70% Train, 15% Validation, 15% Test
    # First split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['label'], 
        random_state=random_seed
    )
    
    # Split Temp into Validation (50% of Temp = 15% of total) and Test (50% of Temp = 15% of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['label'], 
        random_state=random_seed
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df

# Example block to test it
if __name__ == "__main__":
    from src.config import config
    
    print("Generating splits for CXR dataset...")
    train_df, val_df, test_df = get_cxr_dataframes(config.CXR_DIR, config.RANDOM_SEED)
    
    print(f"Total images found: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print("\\nClass distribution in Train:")
    print(train_df['label'].value_counts(normalize=True))
