# Demo Guide - DAIA

This guide explains how to use the DAIA demo interfaces after training your model.

## 🎯 Two Demo Options Available

### Option 1: Streamlit App (RECOMMENDED) ⭐
**Advanced dashboard with statistics and full features**

### Option 2: Gradio Demo
**Simple, quick interface**

---

## 🚀 Quick Start

### After Training Your Model

Once you've run `python src/train.py`, you'll have:
- ✅ Trained model: `models/best_model.pth`
- ✅ Training curves: `outputs/plots/training_curves.png`
- ✅ Confusion matrix: `outputs/plots/confusion_matrix.png`
- ✅ Training logs: `outputs/logs/training.log`

### Launch the Demo

**Option 1: Streamlit (Full Featured)**
```bash
# Launch Streamlit app
streamlit run src/app.py

# Or use the launcher
python launch_demo.py --interface streamlit
```

**Option 2: Gradio (Simple)**
```bash
# Launch Gradio demo
python src/demo.py

# Or use the launcher
python launch_demo.py --interface gradio
```

---

## 📊 Streamlit App Features

### 1. **Image Analysis Tab** 🔍

**Upload an image and get:**
- ✅ Prediction (Real / AI-Generated)
- ✅ Confidence score
- ✅ Risk assessment (High/Medium/Low)
- ✅ Detailed probability breakdown
- ✅ Visual explanation (XAI heatmap)
- ✅ Recommended action
- ✅ Downloadable JSON report

**How to use:**
1. Click "Browse files" or drag & drop an image
2. Wait for analysis (< 1 second)
3. Review the prediction and explanation
4. Download report if needed

### 2. **Model Statistics Tab** 📊

**View comprehensive model performance:**

**Training Progress:**
- Training and validation loss curves
- Training and validation accuracy curves
- How to interpret overfitting/underfitting

**Confusion Matrix:**
- Visual representation of predictions
- True Positives, False Positives, etc.
- Precision, Recall, F1-Score

**Model Architecture:**
- Base model information (ViT)
- Parameter count
- XAI method used

**Dataset Information:**
- Total images used
- Real vs AI-generated split
- Balance ratio

**Training Logs:**
- Complete training history
- Epoch-by-epoch metrics

### 3. **About Tab** ℹ️

Information about:
- How the system works
- Technology stack
- Limitations
- Use cases

---

## 🎨 Streamlit Interface Guide

### Sidebar Controls

**Model Status:**
- Shows if model is loaded successfully
- Model architecture info

**Training Info:**
- Number of epochs trained
- Available visualizations

**Settings:**
- ☑️ Show XAI Explanation (toggle on/off)
- 🎚️ Confidence Threshold (adjust 0-100%)

### Main Interface

#### Prediction Results

**For AI-Generated Images:**

🔴 **HIGH CONFIDENCE (>90%)**
```
⚠️ AI-GENERATED (HIGH CONFIDENCE)
95.3%
🔴 This image is very likely AI-generated
Recommendation: FLAG FOR MANUAL REVIEW
```

🟡 **MODERATE CONFIDENCE (70-90%)**
```
⚠️ AI-GENERATED (MODERATE CONFIDENCE)
82.1%
🟡 This image is possibly AI-generated
Recommendation: Requires additional verification
```

🟢 **LOW CONFIDENCE (<70%)**
```
⚠️ AI-GENERATED (LOW CONFIDENCE)
65.4%
🟢 Uncertain prediction
Recommendation: Expert review recommended
```

**For Real Images:**

✅ **HIGH CONFIDENCE (>90%)**
```
✅ REAL PHOTO (HIGH CONFIDENCE)
94.2%
This image appears to be an authentic photograph
Status: Likely legitimate claim
```

### XAI Explanation

Two visualizations shown:

1. **Attention Heatmap**
   - Pure heatmap showing attention weights
   - Red/Yellow = High attention (suspicious areas)
   - Blue/Green = Low attention

2. **Overlaid Explanation**
   - Original image with heatmap overlay
   - Shows exactly where model focused
   - Easy to interpret

### Risk Assessment

Automatic risk calculation:
- **Risk Level**: High/Medium/Low
- **Risk Score**: Percentage (0-100%)
- **Recommended Action**: What to do next

---

## 📥 Downloading Results

Click **"Download Analysis Report (JSON)"** to get:

```json
{
  "prediction": "AI-Generated",
  "confidence": 0.953,
  "probabilities": {
    "Real": 0.047,
    "AI-Generated": 0.953
  },
  "timestamp": "2025-11-08T14:30:45",
  "risk_assessment": "Reject claim / Manual investigation required"
}
```

Use this for:
- Record keeping
- Audit trail
- Integration with other systems
- Batch processing logs

---

## 🎮 Gradio Interface (Simple Alternative)

### Features:
- ✅ Image upload
- ✅ Prediction with confidence
- ✅ Visual explanation
- ✅ Simple, clean interface

### How to use:
1. Upload image
2. Click "🔍 Analyze Image"
3. View results instantly

**Best for:**
- Quick testing
- Simple demos
- Non-technical users
- Mobile devices

---

## 💡 Best Practices

### For Presentations/Demos

1. **Start with the Statistics Tab**
   - Show training curves (proves model learned)
   - Show confusion matrix (proves performance)
   - Explain accuracy achieved

2. **Switch to Prediction Tab**
   - Upload a **real** image first → Show correct prediction
   - Upload an **AI** image second → Show detection
   - Highlight the XAI explanation

3. **Explain the Risk Assessment**
   - Show how confidence translates to action
   - Discuss thresholds and trade-offs

### For Testing

1. **Test with known examples**
   - Use images from your test set
   - Verify predictions match training results

2. **Test edge cases**
   - Very damaged cars
   - Different angles
   - Various lighting conditions

3. **Check consistency**
   - Upload same image multiple times
   - Should get identical results

### For Production Use

1. **Set appropriate thresholds**
   - High confidence (>90%) → Auto-process
   - Medium (70-90%) → Flag for review
   - Low (<70%) → Manual inspection

2. **Keep audit logs**
   - Download JSON reports
   - Track decisions over time
   - Monitor for drift

3. **Regular retraining**
   - As new AI generators emerge
   - When accuracy drops
   - Add challenging examples

---

## 🔧 Troubleshooting

### "No trained model found"

**Solution:**
```bash
# Train the model first
python src/train.py

# Then launch demo
streamlit run src/app.py
```

### Demo won't start

**Check dependencies:**
```bash
pip install streamlit gradio
```

### Image upload fails

**Check:**
- File format (JPG, PNG supported)
- File size (< 10MB recommended)
- Image not corrupted

### Slow predictions

**Causes:**
- Running on CPU (normal, ~2-5 seconds)
- Large image (resize to 224x224 helps)
- First prediction (model loading)

**Solutions:**
- Use GPU if available
- Resize images before upload
- Wait for first prediction (subsequent are faster)

### XAI explanation not showing

**Check:**
- "Show XAI Explanation" checkbox in sidebar
- Model loaded successfully
- Sufficient memory available

---

## 📱 Remote Access

### Share with Others

**Streamlit:**
```bash
# Get shareable link (requires Streamlit account)
streamlit run src/app.py --server.address=0.0.0.0
```

**Gradio:**
```python
# In demo.py, set share=True
demo.launch(share=True)  # Creates public link for 72 hours
```

### Access from Phone/Tablet

1. Find your computer's IP address:
   ```bash
   # Linux/Mac
   ifconfig | grep inet
   
   # Windows
   ipconfig
   ```

2. Access from same network:
   ```
   http://<your-ip>:8501  (Streamlit)
   http://<your-ip>:7860  (Gradio)
   ```

---

## 🎓 For Your Presentation

### Demo Flow (5 minutes)

**1. Introduction (30 sec)**
- "I built an AI detection system for insurance fraud"
- "Let me show you how it works"

**2. Show Statistics (1 min)**
- Navigate to "Model Statistics" tab
- Point out training curves
- Highlight accuracy achieved
- Show confusion matrix

**3. Live Prediction - Real Image (1.5 min)**
- Upload a real car damage photo
- Show prediction → "Real Photo"
- Point out high confidence
- Show XAI explanation (what model looked at)

**4. Live Prediction - AI Image (1.5 min)**
- Upload an AI-generated image
- Show prediction → "AI-Generated"
- Highlight risk assessment
- Explain recommended action

**5. Wrap-up (30 sec)**
- "System provides explainable, actionable insights"
- "Can be integrated into insurance workflows"

### Key Points to Emphasize

✅ **Accuracy**: "80-90% accuracy on test set"
✅ **Explainability**: "Not a black box - we can see what it's looking at"
✅ **Speed**: "Less than 1 second per image"
✅ **Actionable**: "Provides clear recommendations"
✅ **Real-world ready**: "Risk assessment for triage"

---

## 📚 Additional Resources

- **README.md**: Project overview
- **QUICKSTART.md**: Step-by-step setup
- **INSTALL.md**: Installation help
- **config.yaml**: Customize settings

---

**You're ready to demo! 🎉**

Command to launch:
```bash
streamlit run src/app.py
```

Then open your browser to: **http://localhost:8501**
