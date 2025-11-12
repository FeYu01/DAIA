"""
Model architecture for DAIA project
Supports Vision Transformer (ViT) and CNN-based models
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from torchvision import models
from typing import Optional


class ViTClassifier(nn.Module):
    """
    Vision Transformer based classifier for AI-generated image detection
    Uses pre-trained ViT with custom classification head
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone initially
        """
        super(ViTClassifier, self).__init__()
        
        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.vit.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # For attention map extraction (XAI)
        self.attention_weights = None
        
    def forward(self, pixel_values):
        """
        Forward pass
        
        Args:
            pixel_values: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values, output_attentions=True)
        
        # Use [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0]  # (B, hidden_size)
        
        # Store attention weights for XAI
        self.attention_weights = outputs.attentions  # List of (B, num_heads, seq_len, seq_len)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def unfreeze_backbone(self):
        """Unfreeze the ViT backbone for fine-tuning"""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("ViT backbone unfrozen")
    
    def get_attention_maps(self, layer: int = -1):
        """
        Get attention maps from specified layer
        
        Args:
            layer: Which transformer layer (-1 for last layer)
            
        Returns:
            Attention weights tensor
        """
        if self.attention_weights is None:
            return None
        return self.attention_weights[layer]


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for AI-generated image detection
    Alternative to ViT for comparison
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: ResNet variant (resnet18, resnet34, resnet50, etc.)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone initially
        """
        super(ResNetClassifier, self).__init__()
        
        # Load pre-trained ResNet
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features
        num_features = self.backbone.fc.in_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("ResNet backbone unfrozen")


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier
    Good balance between accuracy and efficiency
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: EfficientNet variant
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pre-trained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze the backbone initially
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("EfficientNet backbone unfrozen")


def create_model(config: dict, device: torch.device):
    """
    Factory function to create model based on config
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Model instance and processor (if applicable)
    """
    model_config = config['model']
    model_name = model_config['name'].lower()
    num_classes = model_config['num_classes']
    dropout = model_config['dropout']
    freeze_backbone = model_config['freeze_backbone']
    
    processor = None
    
    if model_name == "vit":
        # Vision Transformer
        pretrained_model = model_config['pretrained_model']
        model = ViTClassifier(
            model_name=pretrained_model,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        # Load processor for ViT
        processor = ViTImageProcessor.from_pretrained(pretrained_model)
        print(f"Created ViT model: {pretrained_model}")
        
    elif model_name in ["resnet", "resnet50", "resnet18", "resnet34"]:
        # ResNet
        model = ResNetClassifier(
            model_name="resnet50" if model_name == "resnet" else model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        print(f"Created ResNet model")
        
    elif model_name in ["efficientnet", "efficientnet_b0"]:
        # EfficientNet
        model = EfficientNetClassifier(
            model_name="efficientnet_b0",
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        print(f"Created EfficientNet model")
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, processor


if __name__ == "__main__":
    # Test model creation
    from utils import load_config, get_device
    
    config = load_config("config.yaml")
    device = get_device()
    
    print("Testing ViT model creation...")
    model, processor = create_model(config, device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test attention extraction
    if hasattr(model, 'get_attention_maps'):
        attention = model.get_attention_maps()
        if attention is not None:
            print(f"Attention shape: {attention.shape}")
