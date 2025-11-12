"""
Interactive Web Demo for DAIA
Uses Gradio for simple web interface
"""

import os
import gradio as gr
import torch
import numpy as np
from PIL import Image

from predict import DAIAPredictor
from utils import load_config


# Initialize predictor globally
print("Initializing DAIA predictor...")
config = load_config("config.yaml")

# Check if model exists
model_path = "models/best_model.pth"
if not os.path.exists(model_path):
    model_path = "models/final_model.pth"
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Please train the model first using: python src/train.py")
        predictor = None
    else:
        predictor = DAIAPredictor(model_path, use_xai=True)
else:
    predictor = DAIAPredictor(model_path, use_xai=True)


def predict_image(image):
    """
    Predict if image is AI-generated
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Tuple of (label, confidence_dict, explanation_image)
    """
    if predictor is None:
        return "Error: No model loaded", {}, None
    
    # Save temporary image
    temp_path = "temp_image.jpg"
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_path)
    else:
        image.save(temp_path)
    
    # Predict
    try:
        result = predictor.predict(
            temp_path,
            return_explanation=True,
            save_explanation=False
        )
        
        # Format output
        label = result['predicted_label']
        confidence = result['confidence']
        
        # Confidence scores for both classes
        confidence_dict = {
            "Real Photo": result['probabilities']['Real'],
            "AI-Generated": result['probabilities']['AI-Generated']
        }
        
        # Get explanation image
        if 'explanation' in result:
            overlaid = result['explanation']['overlaid']
            # Convert to 0-255 range for display
            explanation_img = (overlaid * 255).astype(np.uint8)
        else:
            explanation_img = None
        
        # Create result text
        if label == "AI-Generated":
            if confidence > 0.9:
                verdict = f"⚠️ **{label}** (HIGH CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "🔴 This image is very likely AI-generated.\n"
                verdict += "**Recommendation**: Flag for manual review in insurance claim."
            elif confidence > 0.7:
                verdict = f"⚠️ **{label}** (MODERATE CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "🟡 This image is possibly AI-generated.\n"
                verdict += "**Recommendation**: Requires additional verification."
            else:
                verdict = f"⚠️ **{label}** (LOW CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "🟢 Uncertain prediction.\n"
                verdict += "**Recommendation**: Expert review recommended."
        else:
            if confidence > 0.9:
                verdict = f"✅ **{label}** (HIGH CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "This image appears to be an authentic photograph."
            elif confidence > 0.7:
                verdict = f"✅ **{label}** (MODERATE CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "This image is probably an authentic photograph."
            else:
                verdict = f"⚠️ **{label}** (LOW CONFIDENCE: {confidence:.1%})\n\n"
                verdict += "Uncertain prediction. May require additional review."
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return verdict, confidence_dict, explanation_img
        
    except Exception as e:
        return f"Error: {str(e)}", {}, None


# Create Gradio interface
with gr.Blocks(title="DAIA - AI Image Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 DAIA - AI-Generated Image Detection
    ## For Insurance Fraud Detection in Automobile Industry
    
    Upload a car damage image to check if it's AI-generated or a real photograph.
    The system will provide an explanation highlighting suspicious areas.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Car Damage Image",
                type="pil",
                height=400
            )
            
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("""
            ### Tips:
            - Upload clear images of car damage
            - Supported formats: JPG, PNG
            - Best results with 224x224 or larger images
            """)
        
        with gr.Column():
            output_text = gr.Markdown(label="Prediction Result")
            
            confidence_plot = gr.Label(
                label="Confidence Scores",
                num_top_classes=2
            )
            
            explanation_image = gr.Image(
                label="Explanation (What the model is looking at)",
                height=400
            )
    
    # Example images (if available)
    gr.Markdown("### Example Images (Click to try)")
    gr.Examples(
        examples=[
            ["data/real/example1.jpg"] if os.path.exists("data/real/example1.jpg") else None,
            ["data/ai_generated/example1.jpg"] if os.path.exists("data/ai_generated/example1.jpg") else None,
        ],
        inputs=input_image,
        label="Sample Images"
    )
    
    gr.Markdown("""
    ---
    ### About This System
    
    **DAIA** uses a Vision Transformer (ViT) model fine-tuned on car damage images to detect AI-generated content.
    
    **How it works**:
    1. **Image Analysis**: The model examines texture patterns, lighting, and damage physics
    2. **Classification**: Predicts if the image is Real or AI-generated
    3. **Explanation**: Highlights suspicious regions using attention mechanisms
    
    **Use Cases**:
    - Insurance claim verification
    - Fraud detection
    - Authenticity assessment
    
    **Limitations**:
    - Trained on specific AI generators (may not detect all types)
    - Requires clear, well-lit images
    - Not 100% accurate - use as a screening tool
    
    ---
    *Built with ❤️ for secure insurance claims*
    """)
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_text, confidence_plot, explanation_image]
    )


# Launch demo
if __name__ == "__main__":
    if predictor is None:
        print("\n" + "="*60)
        print("⚠️  WARNING: No trained model found!")
        print("="*60)
        print("\nPlease train a model first:")
        print("  1. Prepare your dataset in data/real/ and data/ai_generated/")
        print("  2. Run: python src/train.py")
        print("  3. Then run this demo again")
        print("\nLaunching demo anyway (will show errors)...\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
