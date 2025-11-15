"""
Data loading and preprocessing for DAIA project
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CarDamageDataset(Dataset):
    """
    Custom Dataset for Car Damage Images (Real vs AI-generated)
    
    Expected folder structure:
    data/
        real/
            image1.jpg
            image2.jpg
            ...
        ai_generated/
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        processor=None
    ):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=Real, 1=AI-generated)
            transform: Albumentations transforms
            processor: HuggingFace processor (for ViT)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.processor = processor
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image_np)
            image_np = transformed['image']
        
        # Use HuggingFace processor if provided (for ViT)
        if self.processor:
            # Processor expects PIL Image
            image_pil = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
            inputs = self.processor(images=image_pil, return_tensors="pt")
            image_tensor = inputs['pixel_values'].squeeze(0)
        else:
            # Convert to tensor manually
            if isinstance(image_np, np.ndarray):
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = image_np
        
        label = self.labels[idx]
        
        return image_tensor, label, img_path


def get_transforms(config: dict, is_training: bool = True):
    """
    Get albumentations transforms based on config
    
    For 32x32 images: Directly resize to 224x224 using high-quality cubic interpolation
    
    Args:
        config: Configuration dictionary
        is_training: Whether this is for training (applies augmentation)
        
    Returns:
        Albumentations compose transform
    """
    image_size = config['data']['image_size']
    aug_config = config['data']['augmentation']
    
    # ViT requires 224x224, so we upsample from 32x32
    target_size = 224
    
    if is_training and aug_config['enabled']:
        # Training transforms with light augmentation
        transform = A.Compose([
            A.Resize(height=target_size, width=target_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-5, 5), p=0.3),
            A.ColorJitter(
                brightness=aug_config['brightness'],
                contrast=aug_config['contrast'],
                saturation=aug_config['saturation'],
                hue=aug_config['hue'],
                p=0.4
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), mean=0, p=0.2),
        ])
    else:
        # Validation/Test transforms: simple resize
        transform = A.Compose([
            A.Resize(height=target_size, width=target_size, interpolation=cv2.INTER_CUBIC),
        ])
    
    return transform


def load_dataset(data_dir: str, config: dict) -> Tuple[List[str], List[int]]:
    """
    Load dataset from directory structure
    
    Supports two structures:
    1. Pre-split: data_dir/train/REAL, data_dir/train/FAKE, data_dir/test/REAL, data_dir/test/FAKE
    2. Combined: data_dir/real, data_dir/ai_generated
    
    Args:
        data_dir: Root data directory
        config: Configuration dictionary
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Debug: Print data directory
    print(f"DEBUG: Checking data_dir = {data_dir}")
    print(f"DEBUG: Directory exists? {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print(f"DEBUG: Contents of {data_dir}:")
        for item in os.listdir(data_dir):
            print(f"  - {item}")
    
    # Check if pre-split structure exists (train/test folders)
    # Special case: Google Drive created "REAL (1)" for train folder
    train_real = os.path.join(data_dir, 'train', 'REAL')
    if not os.path.exists(train_real):
        train_real = os.path.join(data_dir, 'train', 'REAL (1)')
    
    train_fake = os.path.join(data_dir, 'train', 'FAKE')
    test_real = os.path.join(data_dir, 'test', 'REAL')
    test_fake = os.path.join(data_dir, 'test', 'FAKE')
    
    print(f"DEBUG: Checking folders:")
    print(f"  train_real: {train_real} -> exists={os.path.exists(train_real)}")
    print(f"  train_fake: {train_fake} -> exists={os.path.exists(train_fake)}")
    print(f"  test_real: {test_real} -> exists={os.path.exists(test_real)}")
    print(f"  test_fake: {test_fake} -> exists={os.path.exists(test_fake)}")
    
    has_presplit = all(os.path.exists(p) for p in [train_real, train_fake, test_real, test_fake])
    
    if has_presplit:
        # Load from pre-split structure
        print("Detected pre-split train/test structure")
        
        # Load train REAL (label=0)
        for img_file in os.listdir(train_real):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(train_real, img_file))
                labels.append(0)
        
        # Load train FAKE (label=1)
        for img_file in os.listdir(train_fake):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(train_fake, img_file))
                labels.append(1)
        
        # Load test REAL (label=0)
        for img_file in os.listdir(test_real):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(test_real, img_file))
                labels.append(0)
        
        # Load test FAKE (label=1)
        for img_file in os.listdir(test_fake):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(test_fake, img_file))
                labels.append(1)
    else:
        # Load from combined structure (old behavior)
        print("Using combined real/ai_generated structure")
        
        # Load real images (label=0)
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(real_dir, img_file))
                    labels.append(0)  # Real
        
        # Load AI-generated images (label=1)
        ai_dir = os.path.join(data_dir, 'ai_generated')
        if os.path.exists(ai_dir):
            for img_file in os.listdir(ai_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(ai_dir, img_file))
                    labels.append(1)  # AI-generated
    
    print(f"Loaded {len(image_paths)} images:")
    print(f"  - Real: {labels.count(0)}")
    print(f"  - AI-generated/FAKE: {labels.count(1)}")
    
    return image_paths, labels


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_split,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: train vs val
    val_relative_size = val_split / (train_split + val_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_relative_size,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"    - Real: {train_labels.count(0)}, AI: {train_labels.count(1)}")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"    - Real: {val_labels.count(0)}, AI: {val_labels.count(1)}")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    print(f"    - Real: {test_labels.count(0)}, AI: {test_labels.count(1)}")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def create_dataloaders(
    config: dict,
    processor=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
        processor: HuggingFace processor (optional, for ViT)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Load dataset
    image_paths, labels = load_dataset(data_dir, config)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Please organize your data as:\n"
                        f"  {data_dir}/real/\n"
                        f"  {data_dir}/ai_generated/")
    
    # Split dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths, labels,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        random_seed=config['seed']
    )
    
    # Get transforms
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Create datasets
    train_dataset = CarDamageDataset(train_paths, train_labels, train_transform, processor)
    val_dataset = CarDamageDataset(val_paths, val_labels, val_transform, processor)
    test_dataset = CarDamageDataset(test_paths, test_labels, val_transform, processor)
    
    # Create dataloaders
    # Disable pin_memory for CPU training to reduce memory usage
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from utils import load_config
    
    config = load_config("config.yaml")
    
    print("Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Test batch
        images, labels, paths = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"First image path: {paths[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your data is organized as:")
        print("  data/")
        print("    real/")
        print("      image1.jpg")
        print("    ai_generated/")
        print("      image1.jpg")
