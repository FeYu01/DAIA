"""
Streamlit Web App for DAIA
Advanced demo with statistics dashboard and image analysis
"""

import streamlit as st
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from predict import DAIAPredictor
from utils import load_config


# Page configuration
st.set_page_config(
    page_title="DAIA - AI Image Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — improved contrast for readability
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f4f72; /* deeper blue for contrast */
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333333; /* darker gray for readability */
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #0b2330; /* dark text for contrast */
    }
    .metric-card h4, .metric-card p {
        color: #0b2330;
        margin: 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        color: #5f3d06; /* dark brown text */
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        color: #145214; /* dark green text */
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        color: #6b0217; /* dark red text */
    }
    /* Buttons and captions - ensure visible on light backgrounds */
    .stButton>button {
        color: #ffffff !important;
    }
    .stCaption, .stText, .stMarkdown {
        color: #222222;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the model predictor (cached)"""
    config = load_config("config.yaml")
    
    # Find model
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "models/final_model.pth"
        if not os.path.exists(model_path):
            return None, "No trained model found. Please train the model first."
    
    try:
        predictor = DAIAPredictor(model_path, use_xai=True)
        return predictor, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


@st.cache_data
def load_training_stats():
    """Load training statistics if available"""
    stats = {}
    
    # Load training log
    log_path = "outputs/logs/training.log"
    if os.path.exists(log_path):
        stats['log_available'] = True
        # Parse log for key metrics (simplified)
        with open(log_path, 'r') as f:
            lines = f.readlines()
            stats['total_epochs'] = len([l for l in lines if 'Epoch' in l and 'Train Loss' in l])
    else:
        stats['log_available'] = False
    
    # Check for plots
    stats['training_curves'] = os.path.exists("outputs/plots/training_curves.png")
    stats['confusion_matrix'] = os.path.exists("outputs/plots/confusion_matrix.png")
    
    return stats


def render_sidebar():
    """Render sidebar with model info and stats"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/car-damage.png", width=100)
        st.title("🚗 DAIA System")
        st.markdown("---")
        
        # Model status
        st.subheader("📊 Model Status")
        
        predictor, error = load_predictor()
        if predictor:
            st.success("✅ Model Loaded")
            st.info(f"**Architecture:** Vision Transformer (ViT)")
            st.info(f"**Classes:** Real / AI-Generated")
        else:
            st.error(f"❌ {error}")
            st.warning("Train the model first:\n```\npython src/train.py\n```")
        
        st.markdown("---")
        
        # Training Statistics
        st.subheader("📈 Training Info")
        stats = load_training_stats()
        
        if stats.get('log_available'):
            st.metric("Total Epochs", stats.get('total_epochs', 'N/A'))
            
            if stats.get('training_curves'):
                st.success("✓ Training curves available")
            if stats.get('confusion_matrix'):
                st.success("✓ Confusion matrix available")
        else:
            st.info("No training history found")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        show_explanation = st.checkbox("Show XAI Explanation", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        
        st.markdown("---")
        st.caption("Built for insurance fraud detection")
        
        return predictor, show_explanation, confidence_threshold


def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">🚗 DAIA - AI Image Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detecting AI-Generated Images for Insurance Fraud Prevention</div>', unsafe_allow_html=True)


def render_statistics_tab(predictor):
    """Render statistics and model performance tab"""
    st.header("📊 Model Performance Statistics")
    
    stats = load_training_stats()
    
    # Training Curves
    if stats.get('training_curves'):
        st.subheader("📈 Training Progress")
        col1, col2 = st.columns(2)
        
        with col1:
            training_curves = Image.open("outputs/plots/training_curves.png")
            st.image(training_curves, caption="Training and Validation Curves", use_column_width=True)
        
        with col2:
            st.markdown("""
            #### How to Interpret:
            
            **Loss Curves (Left)**:
            - 📉 Should decrease over time
            - ✅ Train and Val should be close
            - ⚠️ If Val > Train significantly = overfitting
            
            **Accuracy Curves (Right)**:
            - 📈 Should increase over time
            - ✅ Val accuracy 80-90% is good
            - 🎯 Higher is better
            """)
    
    st.markdown("---")
    
    # Confusion Matrix
    if stats.get('confusion_matrix'):
        st.subheader("🎯 Confusion Matrix")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            confusion = Image.open("outputs/plots/confusion_matrix.png")
            st.image(confusion, caption="Test Set Confusion Matrix", use_column_width=True)
        
        with col2:
            st.markdown("""
            #### How to Read:
            
            **Diagonal (Top-left to bottom-right)**:
            - ✅ Correct predictions
            - Higher values = better
            
            **Off-diagonal**:
            - ❌ Incorrect predictions
            - Lower values = better
            
            **Ideal**: Dark diagonal, light off-diagonal
            
            #### Metrics:
            - **Accuracy** = Total correct / Total predictions
            - **Precision** = How many predicted AI are actually AI
            - **Recall** = How many actual AI we detected
            - **F1-Score** = Harmonic mean of Precision & Recall
            """)
    
    st.markdown("---")
    
    # Model Information
    st.subheader("🔧 Model Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Base Model</h4>
        <p>Vision Transformer (ViT)</p>
        <p><b>google/vit-base-patch16-224-in21k</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Parameters</h4>
        <p>~86 Million total</p>
        <p>Pre-trained on ImageNet-21k</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>XAI Method</h4>
        <p>Attention Rollout</p>
        <p>Native ViT visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Info
    st.markdown("---")
    st.subheader("📁 Dataset Information")
    
    # Count images in dataset
    real_count = len([f for f in os.listdir("data/real") if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists("data/real") else 0
    ai_count = len([f for f in os.listdir("data/ai_generated") if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists("data/ai_generated") else 0
    total_count = real_count + ai_count
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", total_count)
    col2.metric("Real Photos", real_count)
    col3.metric("AI-Generated", ai_count)
    
    if total_count > 0:
        balance = min(real_count, ai_count) / max(real_count, ai_count) if max(real_count, ai_count) > 0 else 0
        col4.metric("Balance Ratio", f"{balance:.2%}")
    
    # Training Log
    if stats.get('log_available'):
        st.markdown("---")
        st.subheader("📝 Training Log")
        with st.expander("View Training Log"):
            with open("outputs/logs/training.log", 'r') as f:
                log_content = f.read()
                st.code(log_content[-3000:], language="text")  # Last 3000 chars


def render_prediction_tab(predictor, show_explanation, confidence_threshold):
    """Render image upload and prediction tab"""
    st.header("🔍 Image Analysis")
    
    if predictor is None:
        st.error("⚠️ Model not loaded. Please train the model first.")
        st.code("python src/train.py", language="bash")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a car damage image to analyze",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of car damage"
    )
    
    # Example images
    st.markdown("### Or try an example:")
    col1, col2, col3 = st.columns(3)
    
    example_clicked = None
    with col1:
        if st.button("📸 Example Real Image"):
            # Try to find an example
            real_images = [f for f in os.listdir("data/real") if f.endswith(('.jpg', '.jpeg', '.png'))][:1]
            if real_images:
                example_clicked = os.path.join("data/real", real_images[0])
    
    with col2:
        if st.button("🤖 Example AI Image"):
            ai_images = [f for f in os.listdir("data/ai_generated") if f.endswith(('.jpg', '.jpeg', '.png'))][:1]
            if ai_images:
                example_clicked = os.path.join("data/ai_generated", ai_images[0])
    
    # Process image
    image_to_process = None
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = "temp_upload.jpg"
        image = Image.open(uploaded_file)
        image.save(temp_path)
        image_to_process = temp_path
    elif example_clicked:
        image_to_process = example_clicked
    
    if image_to_process:
        st.markdown("---")
        
        # Show original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            original_img = Image.open(image_to_process)
            st.image(original_img, use_column_width=True)
        
        with col2:
            st.subheader("🔬 Analysis")
            
            # Predict
            with st.spinner("Analyzing image..."):
                result = predictor.predict(
                    image_to_process,
                    return_explanation=show_explanation,
                    save_explanation=False
                )
            
            # Display results
            label = result['predicted_label']
            confidence = result['confidence']
            
            # Verdict box
            if label == "AI-Generated":
                if confidence > 0.9:
                    st.markdown(f"""
                    <div class="danger-box">
                    <h3>⚠️ AI-GENERATED (HIGH CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>🔴 This image is very likely AI-generated</b></p>
                    <p><b>Recommendation:</b> FLAG FOR MANUAL REVIEW</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif confidence > confidence_threshold:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h3>⚠️ AI-GENERATED (MODERATE CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>🟡 This image is possibly AI-generated</b></p>
                    <p><b>Recommendation:</b> Requires additional verification</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h3>⚠️ AI-GENERATED (LOW CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>🟢 Uncertain prediction</b></p>
                    <p><b>Recommendation:</b> Expert review recommended</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                if confidence > 0.9:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ REAL PHOTO (HIGH CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>This image appears to be an authentic photograph</b></p>
                    <p><b>Status:</b> Likely legitimate claim</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif confidence > confidence_threshold:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ REAL PHOTO (MODERATE CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>This image is probably authentic</b></p>
                    <p><b>Status:</b> Proceed with normal processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h3>✅ REAL PHOTO (LOW CONFIDENCE)</h3>
                    <h2>{confidence:.1%}</h2>
                    <p><b>⚠️ Uncertain prediction</b></p>
                    <p><b>Recommendation:</b> May require additional review</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("---")
        st.subheader("📊 Detailed Prediction Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability chart
            probs = result['probabilities']
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = list(probs.keys())
            values = list(probs.values())
            colors = ['#28a745', '#dc3545']
            
            bars = ax.barh(classes, values, color=colors)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.set_title('Class Probabilities')
            
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.02, i, f'{val:.1%}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Risk assessment
            st.markdown("#### 🎯 Risk Assessment")
            
            if label == "AI-Generated":
                if confidence > 0.9:
                    risk_level = "🔴 HIGH RISK"
                    risk_score = confidence * 100
                    action = "Reject claim / Manual investigation required"
                elif confidence > confidence_threshold:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_score = confidence * 100
                    action = "Request additional documentation"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_score = confidence * 100
                    action = "Proceed with caution"
            else:
                risk_level = "✅ LOW RISK"
                risk_score = (1 - confidence) * 100
                action = "Proceed with claim processing"
            
            st.metric("Risk Level", risk_level)
            st.progress(min(risk_score / 100, 1.0))
            st.info(f"**Recommended Action:** {action}")
            
            # Timestamp
            st.caption(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # XAI Explanation
        if show_explanation and 'explanation' in result:
            st.markdown("---")
            st.subheader("🧠 AI Explanation - What the Model is Looking At")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Attention Heatmap**")
                st.caption("Warmer colors (red) = Higher attention")
                heatmap = (result['explanation']['heatmap'] * 255).astype(np.uint8)
                st.image(heatmap, use_column_width=True)
            
            with col2:
                st.markdown("**Overlaid Explanation**")
                st.caption("Suspicious regions highlighted")
                overlaid = (result['explanation']['overlaid'] * 255).astype(np.uint8)
                st.image(overlaid, use_column_width=True)
            
            st.info("""
            **How to interpret the explanation:**
            - 🔴 **Red/Yellow regions**: Areas the model focused on for its decision
            - 🔵 **Blue/Green regions**: Less influential areas
            - The model examines texture patterns, lighting consistency, and damage physics
            """)
        
        # Download results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare JSON report
            report = {
                "prediction": label,
                "confidence": float(confidence),
                "probabilities": {k: float(v) for k, v in result['probabilities'].items()},
                "timestamp": datetime.now().isoformat(),
                "risk_assessment": action
            }
            
            st.download_button(
                "📥 Download Analysis Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Clean up
        if uploaded_file and os.path.exists("temp_upload.jpg"):
            os.remove("temp_upload.jpg")


def main():
    """Main application"""
    render_header()
    
    # Sidebar
    predictor, show_explanation, confidence_threshold = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Image Analysis", "📊 Model Statistics", "ℹ️ About"])
    
    with tab1:
        render_prediction_tab(predictor, show_explanation, confidence_threshold)
    
    with tab2:
        render_statistics_tab(predictor)
    
    with tab3:
        st.header("ℹ️ About DAIA")
        
        st.markdown("""
        ### 🚗 DAIA - AI Image Detection System
        
        **Purpose**: Detect AI-generated car damage images for insurance fraud prevention
        
        #### How It Works:
        
        1. **Image Upload**: Upload a car damage photo
        2. **AI Analysis**: Vision Transformer (ViT) analyzes the image
        3. **Classification**: Determines if image is Real or AI-generated
        4. **Explanation**: Shows which parts influenced the decision (XAI)
        5. **Risk Assessment**: Provides fraud risk evaluation
        
        #### Key Features:
        
        - ✅ **High Accuracy**: 80-90% accuracy on test data
        - ✅ **Explainable AI**: Visual explanations for decisions
        - ✅ **Fast Processing**: < 1 second per image
        - ✅ **Transfer Learning**: Leverages ImageNet pre-training
        - ✅ **Domain-Specific**: Trained on car damage images
        
        #### Technology Stack:
        
        - **Model**: Vision Transformer (ViT)
        - **Framework**: PyTorch, HuggingFace Transformers
        - **XAI**: Attention Rollout
        - **UI**: Streamlit
        
        #### Limitations:
        
        - Trained on specific AI generators (may not detect all)
        - Requires clear, well-lit images
        - Not 100% accurate - use as screening tool
        - Should be combined with manual review for final decisions
        
        #### Use Cases:
        
        - 🔍 Insurance claim verification
        - 🚨 Fraud detection screening
        - ✅ Authenticity assessment
        - 📋 Quality control
        
        ---
        
        **Built with ❤️ for secure insurance claims**
        
        For more information, see the [GitHub Repository](https://github.com/FeYu01/DAIA)
        """)


if __name__ == "__main__":
    main()
