"""
Training script for DAIA project
Implements training loop with validation, metrics tracking, and model checkpointing
"""

import sys
import shutil

# Preflight dependency check: detect common missing packages and give actionable guidance.
def _preflight_check():
    missing = []
    for pkg in ('transformers', 'torch', 'albumentations'):
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)

    if missing:
        print('\nERROR: Missing Python packages detected:', ', '.join(missing))
        print('Current Python executable:', sys.executable)
        print('\nRecommended fixes:')
        print(' 1) Activate the project virtualenv and run the script:')
        print('      source venv/bin/activate')
        print('      python src/train.py')
        print('\n 2) Or run using the project venv interpreter directly:')
        print('      /workspaces/DAIA/venv/bin/python src/train.py')
        print('\n 3) Or install the missing packages into the current interpreter:')
        print('      pip install ' + ' '.join(missing))
        print('\nIf you want, I can generate a `requirements.txt` from the venv for reproducible installs.')
        sys.exit(1)


_preflight_check()

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environments

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns

from utils import (
    load_config, set_seed, setup_logging, get_device,
    create_directories, count_parameters, format_time, save_checkpoint
)
from model import create_model
from data_loader import create_dataloaders


class Trainer:
    """
    Trainer class for model training and evaluation
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Setup
        set_seed(config['seed'])
        self.device = get_device()
        self.logger = setup_logging(
            config['paths']['log_dir'],
            config['logging']['level']
        )
        create_directories(config)
        
        # Create model
        self.logger.info("Creating model...")
        self.model, self.processor = create_model(config, self.device)
        
        # Create dataloaders
        self.logger.info("Loading dataset...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, self.processor
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.early_stopping_counter = 0
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        train_config = self.config['training']
        lr = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        
        if train_config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif train_config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        train_config = self.config['training']
        scheduler_type = train_config['scheduler']
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs']
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=train_config['epochs'] // 3,
                gamma=0.1
            )
        elif scheduler_type == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """
        Validate on validation set
        
        Returns:
            Tuple of (average_loss, accuracy, precision, recall, f1)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]  ")
        
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self):
        """
        Main training loop
        """
        train_config = self.config['training']
        num_epochs = train_config['epochs']
        early_stopping_patience = train_config['early_stopping_patience']
        freeze_epochs = self.config['model']['freeze_epochs']
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Unfreeze backbone after initial epochs
            if epoch == freeze_epochs and self.config['model']['freeze_backbone']:
                self.logger.info(f"Unfreezing backbone at epoch {epoch + 1}")
                self.model.unfreeze_backbone()
                # Recreate optimizer for unfrozen parameters
                self.optimizer = self._create_optimizer()
                self.scheduler = self._create_scheduler()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate step
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stopping_counter = 0
                
                if train_config['save_best_only']:
                    model_path = os.path.join(
                        self.config['paths']['model_save_dir'],
                        'best_model.pth'
                    )
                    torch.save(self.model.state_dict(), model_path)
                    self.logger.info(f"✓ Best model saved (Val Acc: {val_acc:.4f})")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {early_stopping_patience} epochs)"
                )
                break
        
        # Training complete
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Total time: {format_time(total_time)}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.logger.info("=" * 60)
        
        # Save final model
        final_model_path = os.path.join(
            self.config['paths']['model_save_dir'],
            'final_model.pth'
        )
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Evaluate on test set
        self.test()
    
    @torch.no_grad()
    def test(self):
        """
        Evaluate on test set
        """
        self.logger.info("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(
            self.config['paths']['model_save_dir'],
            'best_model.pth'
        )
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info("Loaded best model for testing")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels, _ in tqdm(self.test_loader, desc="Testing"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        self.logger.info("\nTest Results:")
        self.logger.info(f"  Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall:    {recall:.4f}")
        self.logger.info(f"  F1 Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        return accuracy, precision, recall, f1
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curve
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.config['paths']['plot_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Training curves saved to {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(8, 6))
        
        class_names = ['Real', 'AI-Generated']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        
        # Save
        plot_path = os.path.join(self.config['paths']['plot_dir'], 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Confusion matrix saved to {plot_path}")
        plt.close()


def main():
    """Main training function"""
    # Load config
    config = load_config("config.yaml")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
