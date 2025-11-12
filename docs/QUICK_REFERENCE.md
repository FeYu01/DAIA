# Quick Reference - DAIA Commands

## 🚀 After Training - How to Use Your Model

### 1. Launch the Demo (Easiest)

```bash
# Streamlit App (RECOMMENDED)
streamlit run src/app.py
# Opens http://localhost:8501

# Or Gradio Demo
python src/demo.py
# Opens http://localhost:7860
```

### 2. Command Line Prediction

```bash
# Single image
python src/predict.py path/to/image.jpg --save-explanation

# Batch of images
python src/predict.py path/to/folder/ --batch --save-explanation

# Without XAI explanations (faster)
python src/predict.py path/to/image.jpg --no-xai
```

### 3. Python Script Usage

```python
from src.predict import DAIAPredictor

# Load predictor
predictor = DAIAPredictor(
    model_path="models/best_model.pth",
    use_xai=True
)

# Predict single image
result = predictor.predict(
    "test_image.jpg",
    return_explanation=True,
    save_explanation=True
)

# Access results
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

---

## 📊 What You Get

### Streamlit App Features

**Tab 1: Image Analysis**
- Upload image → Get prediction
- Confidence score (0-100%)
- Risk assessment (High/Medium/Low)
- XAI heatmap showing focus areas
- Recommended action
- Download JSON report

**Tab 2: Model Statistics**
- Training/validation curves
- Confusion matrix
- Model architecture info
- Dataset statistics
- Training logs

**Tab 3: About**
- System information
- How it works
- Limitations

### Command Line Output

```
============================================================
Image: damaged_car.jpg
============================================================
⚠️ Prediction: AI-Generated
  Confidence: 94.23%

Class Probabilities:
  Real            [████░░░░░░░░] 5.77%
  AI-Generated    [███████████░] 94.23%

Interpretation:
  🔴 HIGH CONFIDENCE - Very likely AI-generated
  → Recommend manual review for insurance claim
============================================================
```

### Saved Files

**After prediction with `--save-explanation`:**
- `outputs/explanations/image_name_explanation.png`
  - Original image
  - Attention heatmap
  - Overlaid explanation

**After training:**
- `models/best_model.pth` - Best model checkpoint
- `outputs/plots/training_curves.png` - Loss & accuracy
- `outputs/plots/confusion_matrix.png` - Performance
- `outputs/logs/training.log` - Full training history

---

## 🎯 Common Use Cases

### Use Case 1: Demo for Presentation

```bash
# Launch full-featured demo
streamlit run src/app.py

# In the browser:
# 1. Go to "Model Statistics" tab → Show performance
# 2. Go to "Image Analysis" tab → Upload test images
# 3. Show predictions + explanations
```

### Use Case 2: Batch Processing Claims

```bash
# Process entire folder
python src/predict.py data/incoming_claims/ --batch --save-explanation

# Check outputs/explanations/ folder for results
```

### Use Case 3: Integration with System

```python
# In your insurance system code
from src.predict import DAIAPredictor

predictor = DAIAPredictor("models/best_model.pth")

def check_claim_image(image_path):
    result = predictor.predict(image_path)
    
    if result['predicted_label'] == 'AI-Generated':
        if result['confidence'] > 0.9:
            return "REJECT", "High confidence AI-generated"
        else:
            return "REVIEW", "Possible AI-generated"
    else:
        return "APPROVE", "Appears authentic"

# Use in your workflow
status, reason = check_claim_image("claim_12345.jpg")
```

---

## 🔍 Interpreting Results

### Confidence Levels

| Confidence | Meaning | Action |
|------------|---------|--------|
| > 90% | High confidence | Trust the prediction |
| 70-90% | Moderate confidence | Review recommended |
| < 70% | Low confidence | Manual inspection needed |

### Risk Assessment (for AI-Generated predictions)

| Risk | Confidence | Recommended Action |
|------|------------|-------------------|
| 🔴 HIGH | > 90% | Reject claim / Investigation |
| 🟡 MEDIUM | 70-90% | Request additional docs |
| 🟢 LOW | < 70% | Proceed with caution |

### XAI Heatmap Colors

- 🔴 **Red/Yellow**: High attention (suspicious areas)
- 🟡 **Orange**: Medium attention
- 🔵 **Blue/Green**: Low attention (normal areas)

**Look for:**
- Inconsistent textures
- Unnatural lighting
- Physics violations
- Repetitive patterns

---

## 🎓 For Your Presentation

### 5-Minute Demo Script

**1. Open Streamlit app** (30 sec)
```bash
streamlit run src/app.py
```

**2. Show Statistics Tab** (1 min)
- "Here's how well the model performed during training"
- Point to accuracy: "We achieved ~85% accuracy"
- Show confusion matrix: "Low false positives/negatives"

**3. Upload Real Image** (1.5 min)
- Switch to "Image Analysis" tab
- Upload a real car damage photo
- Show: "Correctly identified as Real with 92% confidence"
- Display XAI: "Model focused on realistic damage patterns"

**4. Upload AI Image** (1.5 min)
- Upload an AI-generated image
- Show: "Detected as AI-Generated with 96% confidence"
- Display XAI: "Model identified suspicious texture patterns"
- Show risk assessment: "Flagged for manual review"

**5. Conclusion** (30 sec)
- "System provides fast, explainable fraud detection"
- "Can be integrated into insurance workflows"

---

## 📦 File Locations Reference

```
After training and running demos:

models/
├── best_model.pth          ← Load this for predictions
└── final_model.pth         ← Alternative model

outputs/
├── explanations/           ← XAI visualizations saved here
├── plots/
│   ├── training_curves.png ← Show in presentations
│   └── confusion_matrix.png ← Show in presentations
└── logs/
    └── training.log        ← Full training history

data/
├── real/                   ← Your real images
└── ai_generated/           ← Your AI images
```

---

## 💡 Tips & Tricks

### Speed Up Predictions

```bash
# Disable XAI for faster processing
python src/predict.py image.jpg --no-xai
```

### Adjust Threshold

In Streamlit app:
- Use sidebar slider to adjust confidence threshold
- Higher threshold = fewer false positives
- Lower threshold = catch more AI images

### Best Images for Demo

✅ **Good demo images:**
- Clear, well-lit photos
- Obvious damage visible
- Standard angles (front, side, rear)

❌ **Avoid for demo:**
- Very blurry images
- Extreme angles
- Heavy filters/editing

---

## 🆘 Troubleshooting

**Demo won't start:**
```bash
pip install streamlit gradio
```

**"No model found":**
```bash
# Train the model first
python src/train.py
```

**Slow predictions:**
- Normal on CPU (~2-5 seconds)
- Use smaller images
- First prediction loads model (subsequent faster)

**Import errors:**
```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## 📚 Documentation

- **DEMO_GUIDE.md** - Detailed demo instructions
- **README.md** - Project overview
- **QUICKSTART.md** - Week-by-week guide
- **INSTALL.md** - Installation help

---

**Quick Start Command:**
```bash
streamlit run src/app.py
```

**Then open your browser to:** http://localhost:8501

**That's it! 🎉**
