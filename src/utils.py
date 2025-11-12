"""
Utility functions for DAIA project
"""

import os
import yaml
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Handle relative paths - check if file exists, otherwise try from parent directory
    if not os.path.exists(config_path):
        # Try from parent directory (when running from src/)
        parent_config = os.path.join('..', config_path)
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            # Try from project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root_config = os.path.join(project_root, config_path)
            if os.path.exists(root_config):
                config_path = root_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "./outputs/logs", level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("DAIA")
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    filepath: str
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        best_val_acc: Best validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
) -> tuple:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, optimizer, epoch, best_val_acc)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch}, best val acc: {best_val_acc:.4f}")
    
    return model, optimizer, epoch, best_val_acc
