"""
Prediction and inference script for DAIA project
Performs single-image prediction with XAI explanations
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple

from utils import load_config, get_device
from model import create_model
from explainer import ExplainerViT, analyze_prediction_confidence


class DAIAPredictor:
    """
    Predictor for AI-generated image detection with explanations
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        use_xai: bool = True
    ):
        """
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            use_xai: Whether to generate explanations
        """
        # Load config
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Create model
        print("Loading model...")
        self.model, self.processor = create_model(self.config, self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model.eval()
        
        # Setup explainer
        self.use_xai = use_xai
        if use_xai:
            xai_method = self.config['xai']['method']
            colormap = self.config['xai']['visualization']['colormap']
            self.explainer = ExplainerViT(self.model, self.device, xai_method, colormap)
            print(f"✓ XAI explainer ready (method: {xai_method})")
        
        # Class names
        self.class_names = ["Real", "AI-Generated"]
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (preprocessed_tensor, original_image_array)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Normalize for visualization
        original_normalized = image_np.astype(np.float32) / 255.0
        
        # Preprocess with HuggingFace processor (for ViT)
        if self.processor:
            inputs = self.processor(images=image, return_tensors="pt")
            image_tensor = inputs['pixel_values']
        else:
            # Manual preprocessing (for CNN models)
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor, original_normalized
    
    def predict(
        self,
        image_path: str,
        return_explanation: bool = True,
        save_explanation: bool = False
    ) -> Dict:
        """
        Predict if image is AI-generated with optional explanation
        
        Args:
            image_path: Path to image file
            return_explanation: Whether to generate explanation
            save_explanation: Whether to save explanation visualization
            
        Returns:
            Dictionary containing prediction and explanation
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        # Results
        predicted_class = predicted.item()
        predicted_label = self.class_names[predicted_class]
        confidence_score = confidence.item()
        
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence_score,
            'probabilities': {
                'Real': probabilities[0, 0].item(),
                'AI-Generated': probabilities[0, 1].item()
            }
        }
        
        # Generate explanation
        if return_explanation and self.use_xai:
            print("Generating explanation...")
            
            # Resize original image to match model input
            original_resized = np.array(
                Image.fromarray((original_image * 255).astype(np.uint8)).resize((224, 224))
            ) / 255.0
            
            heatmap, overlaid = self.explainer.generate_explanation(
                image_tensor,
                original_resized,
                target_class=predicted_class
            )
            
            result['explanation'] = {
                'heatmap': heatmap,
                'overlaid': overlaid
            }
            
            # Save explanation if requested
            if save_explanation:
                output_dir = self.config['paths']['explanation_dir']
                self.explainer.save_explanation(
                    image_path,
                    heatmap,
                    overlaid,
                    output_dir,
                    predicted_label,
                    confidence_score
                )
        
        return result
    
    def predict_batch(self, image_dir: str, save_explanations: bool = True) -> list:
        """
        Predict on all images in a directory
        
        Args:
            image_dir: Directory containing images
            save_explanations: Whether to save explanations
            
        Returns:
            List of prediction results
        """
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"Found {len(image_files)} images")
        
        results = []
        for image_path in image_files:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            try:
                result = self.predict(
                    image_path,
                    return_explanation=True,
                    save_explanation=save_explanations
                )
                results.append(result)
                
                # Print result
                self.print_result(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def print_result(self, result: Dict):
        """
        Print prediction result in formatted way
        
        Args:
            result: Prediction result dictionary
        """
        print("\n" + "=" * 60)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print("=" * 60)
        
        # Prediction
        label = result['predicted_label']
        conf = result['confidence']
        
        if label == "AI-Generated":
            symbol = "⚠️ "
            color_start = "\033[91m"  # Red
        else:
            symbol = "✓ "
            color_start = "\033[92m"  # Green
        color_end = "\033[0m"
        
        print(f"{symbol}Prediction: {color_start}{label}{color_end}")
        print(f"  Confidence: {conf:.2%}")
        
        # Probabilities
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {class_name:15s} [{bar}] {prob:.2%}")
        
        # Interpretation
        print("\nInterpretation:")
        if label == "AI-Generated":
            if conf > 0.9:
                print("  🔴 HIGH CONFIDENCE - Very likely AI-generated")
                print("  → Recommend manual review for insurance claim")
            elif conf > 0.7:
                print("  🟡 MODERATE CONFIDENCE - Possibly AI-generated")
                print("  → Flag for additional verification")
            else:
                print("  🟢 LOW CONFIDENCE - Uncertain prediction")
                print("  → May require expert review")
        else:
            if conf > 0.9:
                print("  ✓ HIGH CONFIDENCE - Likely authentic")
            elif conf > 0.7:
                print("  ✓ MODERATE CONFIDENCE - Probably authentic")
            else:
                print("  ⚠️  LOW CONFIDENCE - Uncertain prediction")
        
        print("=" * 60 + "\n")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="DAIA: AI-Generated Image Detection for Insurance Fraud"
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to image file or directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pth',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-xai',
        action='store_true',
        help='Disable XAI explanations'
    )
    parser.add_argument(
        '--save-explanation',
        action='store_true',
        help='Save explanation visualizations'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in directory'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DAIAPredictor(
        model_path=args.model,
        config_path=args.config,
        use_xai=not args.no_xai
    )
    
    # Predict
    if args.batch or os.path.isdir(args.image_path):
        # Batch prediction
        results = predictor.predict_batch(
            args.image_path,
            save_explanations=args.save_explanation
        )
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH PREDICTION SUMMARY")
        print("=" * 60)
        ai_count = sum(1 for r in results if r['predicted_label'] == 'AI-Generated')
        real_count = len(results) - ai_count
        print(f"Total images: {len(results)}")
        print(f"  Real: {real_count}")
        print(f"  AI-Generated: {ai_count}")
        print("=" * 60)
        
    else:
        # Single image prediction
        result = predictor.predict(
            args.image_path,
            return_explanation=True,
            save_explanation=args.save_explanation
        )
        predictor.print_result(result)


if __name__ == "__main__":
    main()
