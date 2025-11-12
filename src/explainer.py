"""
Explainability (XAI) module for DAIA project
Implements Grad-CAM and Attention visualization for ViT
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environments

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformer
    Native XAI method for ViT - no gradients needed
    """
    
    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        """
        Args:
            model: ViT model
            head_fusion: How to combine attention heads ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attentions to discard
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate attention rollout heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            
        Returns:
            Attention mask (H, W)
        """
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Get attention weights from all layers
        attentions = self.model.attention_weights
        
        if attentions is None:
            raise ValueError("Model doesn't return attention weights")
        
        # Number of tokens (patches + CLS token)
        num_tokens = attentions[0].shape[-1]
        
        # Initialize attention matrix
        result = torch.eye(num_tokens).to(input_tensor.device)
        
        for attention in attentions:
            # Combine attention heads
            if self.head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Invalid head_fusion: {self.head_fusion}")
            
            # Take first sample in batch
            attention_heads_fused = attention_heads_fused[0]
            
            # Discard the lowest attentions (optional)
            flat = attention_heads_fused.view(-1)
            _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), largest=False)
            attention_heads_fused.view(-1)[indices] = 0
            
            # Normalize
            I = torch.eye(num_tokens).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            # Matrix multiplication (attention rollout)
            result = torch.matmul(a, result)
        
        # Get CLS token attention to patches
        mask = result[0, 1:]  # Exclude CLS token itself
        
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(grid_size, grid_size).cpu().numpy()
        
        return mask


class ExplainerViT:
    """
    Comprehensive explainer for ViT models
    Supports both Attention Rollout and Grad-CAM
    """
    
    def __init__(
        self,
        model,
        device: torch.device,
        method: str = "attention",
        colormap: str = "jet"
    ):
        """
        Args:
            model: ViT model
            device: Device to run on
            method: Explanation method ('attention' or 'gradcam')
            colormap: Colormap for visualization
        """
        self.model = model
        self.device = device
        self.method = method
        self.colormap = colormap
        
        # Initialize explainer based on method
        if method == "attention":
            self.explainer = ViTAttentionRollout(model)
        elif method == "gradcam":
            # For Grad-CAM, target the last layer norm before classifier
            target_layers = [model.vit.layernorm]
            self.explainer = GradCAM(model=model, target_layers=target_layers)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_explanation(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate explanation heatmap
        
        Args:
            image: Preprocessed image tensor (1, C, H, W)
            original_image: Original image as numpy array (H, W, 3), normalized to [0, 1]
            target_class: Target class for Grad-CAM (None = predicted class)
            
        Returns:
            Tuple of (heatmap, overlaid_image)
        """
        self.model.eval()
        
        if self.method == "attention":
            # Attention Rollout
            mask = self.explainer(image)
            
            # Resize to original image size
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
            
            # Normalize to [0, 1]
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            
        else:  # gradcam
            # Grad-CAM
            if target_class is None:
                # Use predicted class
                with torch.no_grad():
                    outputs = self.model(image)
                    target_class = outputs.argmax(dim=1).item()
            
            grayscale_cam = self.explainer(
                input_tensor=image,
                targets=None  # Will use predicted class
            )
            
            mask = grayscale_cam[0, :]  # Take first batch item
        
        # Apply colormap
        heatmap = self.apply_colormap(mask)
        
        # Overlay on original image
        overlaid = self.overlay_heatmap(original_image, heatmap)
        
        return heatmap, overlaid
    
    def apply_colormap(self, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Apply colormap to mask
        
        Args:
            mask: Grayscale mask (H, W)
            alpha: Transparency
            
        Returns:
            Colored heatmap (H, W, 3)
        """
        # Normalize to 0-255
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply colormap
        if self.colormap == "jet":
            heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        elif self.colormap == "hot":
            heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_HOT)
        elif self.colormap == "viridis":
            heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_VIRIDIS)
        else:
            heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap / 255.0
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on image
        
        Args:
            image: Original image (H, W, 3), normalized to [0, 1]
            heatmap: Heatmap (H, W, 3), normalized to [0, 1]
            alpha: Heatmap transparency (0=invisible, 1=opaque)
            
        Returns:
            Overlaid image (H, W, 3)
        """
        # Ensure same size
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend
        overlaid = (1 - alpha) * image + alpha * heatmap
        
        # Clip to valid range
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid
    
    def save_explanation(
        self,
        image_path: str,
        heatmap: np.ndarray,
        overlaid: np.ndarray,
        output_dir: str,
        prediction: str,
        confidence: float
    ):
        """
        Save explanation visualization
        
        Args:
            image_path: Original image path
            heatmap: Heatmap array
            overlaid: Overlaid image array
            output_dir: Output directory
            prediction: Prediction label
            confidence: Prediction confidence
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = os.path.basename(image_path).split('.')[0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        original = Image.open(image_path).convert('RGB')
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title(f'Attention Heatmap\n({self.method})')
        axes[1].axis('off')
        
        # Overlaid
        axes[2].imshow(overlaid)
        axes[2].set_title(f'Explanation\n{prediction} ({confidence:.1%})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(output_dir, f"{filename}_explanation.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Explanation saved to: {output_path}")


def visualize_attention_map(
    attention_map: np.ndarray,
    original_image: np.ndarray,
    title: str = "Attention Map"
) -> None:
    """
    Visualize attention map
    
    Args:
        attention_map: Attention weights (H, W)
        original_image: Original image
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_prediction_confidence(
    model,
    image: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Analyze prediction with confidence scores
    
    Args:
        model: Model
        image: Input image tensor
        device: Device
        
    Returns:
        Dictionary with prediction analysis
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(image.to(device))
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    
    class_names = ["Real", "AI-Generated"]
    
    analysis = {
        'predicted_class': predicted.item(),
        'predicted_label': class_names[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {
            'Real': probabilities[0, 0].item(),
            'AI-Generated': probabilities[0, 1].item()
        }
    }
    
    return analysis


if __name__ == "__main__":
    # Test explainer
    from model import create_model
    from utils import load_config, get_device
    
    config = load_config("config.yaml")
    device = get_device()
    
    print("Creating model...")
    model, processor = create_model(config, device)
    
    print("Testing explainer...")
    explainer = ExplainerViT(model, device, method="attention")
    
    # Test with dummy image
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_original = np.random.rand(224, 224, 3)
    
    print("Generating explanation...")
    heatmap, overlaid = explainer.generate_explanation(dummy_image, dummy_original)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Overlaid shape: {overlaid.shape}")
    print("✓ Explainer test passed!")
