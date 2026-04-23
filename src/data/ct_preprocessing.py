import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple

try:
    import nibabel as nib
    import cv2
except ImportError:
    print("Please install required libraries: pip install nibabel opencv-python-headless pandas scikit-learn tqdm")

def window_hu(image: np.ndarray, window_min: float = -1000, window_max: float = 400) -> np.ndarray:
    """
    Applies Hounsfield Unit (HU) windowing.
    By default [-1000, 400] is excellent for lung tissue and lesions.
    """
    image = np.clip(image, window_min, window_max)
    # Normalize to [0, 1]
    image = (image - window_min) / (window_max - window_min)
    # Convert to [0, 255] uint8
    return (image * 255.0).astype(np.uint8)

def process_and_extract_slices(
    data_dir: Path, 
    output_dir: Path, 
    target_size: tuple = (256, 256),
    slice_range: tuple = (0.2, 0.8)
) -> pd.DataFrame:
    """
    Reads MosMedData NIfTI files, applies windowing, extracts central slices,
    resizes them, and saves as PNGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = ['CT-0', 'CT-1', 'CT-2', 'CT-3', 'CT-4']
    records = []
    
    for cls in classes:
        cls_dir = data_dir / "studies" / cls
        if not cls_dir.exists():
            continue
            
        # Group CT-3 and CT-4 into CT-3+ as per the plan
        mapped_label = 'CT-3+' if cls in ['CT-3', 'CT-4'] else cls
        
        # Output directory for this class
        out_cls_dir = output_dir / mapped_label
        out_cls_dir.mkdir(exist_ok=True)
        
        nii_files = glob.glob(str(cls_dir / "*.nii*"))
        
        print(f"Processing {len(nii_files)} studies for class {cls} (Mapped to {mapped_label})...")
        for nii_path in tqdm(nii_files):
            study_id = Path(nii_path).stem.replace('.nii', '')
            
            # Load NIfTI
            img = nib.load(nii_path)
            vol = img.get_fdata()
            
            # Apply Windowing
            vol = window_hu(vol)
            
            # Determine slice range along Z-axis (usually the 3rd dimension)
            num_slices = vol.shape[-1]
            start_idx = int(num_slices * slice_range[0])
            end_idx = int(num_slices * slice_range[1])
            
            for z in range(start_idx, end_idx):
                slice_2d = vol[:, :, z]
                
                # Usually NIfTI orientation needs a rotation to look 'upright'
                slice_2d = np.rot90(slice_2d)
                
                # Resize
                slice_2d = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_AREA)
                
                # Save
                slice_filename = f"{study_id}_slice_{z:03d}.png"
                out_filepath = out_cls_dir / slice_filename
                cv2.imwrite(str(out_filepath), slice_2d)
                
                records.append({
                    'study_id': study_id,
                    'image_path': str(out_filepath),
                    'slice_index': z,
                    'total_slices': num_slices,
                    'label': mapped_label
                })
                
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "labels_metadata.csv", index=False)
    return df

def get_ct_dataframes(df: pd.DataFrame, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates train, validation, and test splits for the CT dataset.
    CRITICAL: Split must be done by 'study_id', NOT by individual slices,
    to prevent data leakage (slices from the same patient in train and test).
    """
    # 1. Get unique studies and their labels
    # Assuming one study has only one label
    study_df = df[['study_id', 'label']].drop_duplicates()
    
    # 2. Stratified split by study
    train_studies, temp_studies = train_test_split(
        study_df, test_size=0.30, stratify=study_df['label'], random_state=random_seed
    )
    
    val_studies, test_studies = train_test_split(
        temp_studies, test_size=0.50, stratify=temp_studies['label'], random_state=random_seed
    )
    
    # 3. Map back to slices
    train_df = df[df['study_id'].isin(train_studies['study_id'])].reset_index(drop=True)
    val_df = df[df['study_id'].isin(val_studies['study_id'])].reset_index(drop=True)
    test_df = df[df['study_id'].isin(test_studies['study_id'])].reset_index(drop=True)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    from src.config import config
    
    # Define where the 2D slices will be saved
    PROCESSED_CT_DIR = config.DATA_DIR / "MosMedData_Chest_Scan" / "processed_2d_slices"
    
    print("Step 1: Extracting 2D slices from NIfTI volumes (This might take a while)...")
    df = process_and_extract_slices(
        data_dir=config.CT_DIR,
        output_dir=PROCESSED_CT_DIR,
        target_size=config.CT_IMAGE_SIZE,
        slice_range=(0.2, 0.8) # 20% to 80% as per plan
    )
    
    print("\\nStep 2: Generating stratified splits (by Study ID)...")
    train_df, val_df, test_df = get_ct_dataframes(df, config.RANDOM_SEED)
    
    print(f"Total slices generated: {len(df)}")
    print(f"Train slices: {len(train_df)}")
    print(f"Validation slices: {len(val_df)}")
    print(f"Test slices: {len(test_df)}")
    
    print("\\nClass distribution in Train (Slices):")
    print(train_df['label'].value_counts(normalize=True))
