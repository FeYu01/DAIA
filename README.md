# DAIA - AI-Generated Image Detection for Insurance Fraud

> Detecting AI-generated images in automobile insurance claims using Vision Transformers (ViT)

**University Project** | 1 Week Timeline | Zero Budget

---

## 📊 Current Status

```
Dataset:       932 / 1000-1500 target
├─ Real:       920 images ✅
└─ AI:         12 images (need ~488-738 more)

Environment:   ✅ Configured
Model:         Ready to train (after data collection)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# In GitHub Codespaces (already done!)
source venv/bin/activate
```

### 2. Collect Dataset

**Target:** 500-750 images per class (1000-1500 total)

- ✅ **Real images**: 920 from Kaggle (DONE!)
- 🔄 **AI images**: Generate using Gemini, Bing, Leonardo.AI

### 3. Train Model

```bash
python src/train.py
```

### 4. Launch Demo

```bash
streamlit run src/app.py
```

---

## 📁 Project Structure

```
DAIA/
├── README.md                   # This file
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
│
├── src/                        # Main code
│   ├── train.py               # Training script
│   ├── predict.py             # Make predictions
│   ├── app.py                 # Streamlit dashboard
│   ├── demo.py                # Gradio demo
│   ├── model.py               # ViT architecture
│   ├── data_loader.py         # Dataset handling
│   ├── explainer.py           # XAI (explanations)
│   └── utils.py               # Utilities
│
├── data/                       # Dataset
│   ├── real/                  # 920 real images ✅
│   └── ai_generated/          # 12 AI images (need more)
│
├── scripts/                    # Helper scripts
│   ├── setup.py               # Environment setup
│   ├── crop_gemini_only.py    # Remove Gemini watermarks
│   └── launch_demo.py         # Demo launcher
│
├── docs/                       # Documentation
│   ├── INSTALL.md             # Installation guide
│   ├── QUICKSTART.md          # Week-by-week timeline
│   └── DEMO_GUIDE.md          # Demo usage guide
│
└── outputs/                    # Generated files (auto-created)
    ├── models/                # Saved model checkpoints
    ├── plots/                 # Training curves
    └── logs/                  # Training logs
```

---

## 🎯 Key Features

- **Vision Transformer (ViT)**: State-of-the-art image classification
- **Explainable AI (XAI)**: Visual explanations via Attention Rollout
- **Auto-resizing**: Any image size → 224x224 automatically
- **Data Augmentation**: Improves model robustness
- **Interactive Demos**: Streamlit & Gradio interfaces
- **GitHub Codespaces**: Works in browser, no local setup needed

---

## 📖 Documentation

All documentation moved to `docs/` folder:

- **[Installation Guide](docs/INSTALL.md)** - Setup instructions
- **[Quick Start](docs/QUICKSTART.md)** - Week-by-week timeline
- **[Demo Guide](docs/DEMO_GUIDE.md)** - How to use the web interface

---

## 🛠️ Common Commands

```bash
# Activate environment (always do this first!)
source venv/bin/activate

# Verify dataset
python scripts/setup.py

# Remove Gemini watermarks (when you get new Gemini images)
python scripts/crop_gemini_only.py

# Train model
python src/train.py

# Predict single image
python src/predict.py path/to/image.jpg

# Launch Streamlit dashboard
streamlit run src/app.py

# Launch Gradio demo
python src/demo.py
```

---

## 🧪 Model Details

**Architecture:** Vision Transformer (ViT)
- Base model: `google/vit-base-patch16-224-in21k`
- Input size: 224x224 pixels
- Binary classification: Real vs AI-generated
- XAI: Attention Rollout for visual explanations

**Training:**
- Optimizer: AdamW
- Learning rate: 1e-4 (with cosine schedule)
- Batch size: 16
- Early stopping: Patience 5 epochs
- Data augmentation: Rotation, flip, color jitter

---

## 📝 License

This project is for educational purposes (university project).

---

## 👤 Author

**Felix Yu**  
GitHub: [@FeYu01](https://github.com/FeYu01)

---

## 🆘 Quick Help

**Problem:** Module not found  
**Solution:** Activate environment: `source venv/bin/activate`

**Problem:** No model found  
**Solution:** Train first: `python src/train.py`

**Problem:** Gemini watermark visible  
**Solution:** Run: `python scripts/crop_gemini_only.py`

For detailed help, see [docs/](docs/)

```bash
python src/demo.py
```
Then open: http://localhost:7860

See **DEMO_GUIDE.md** for detailed demo instructions.

---

## 🎨 Demo Screenshots

### Streamlit App Interface

The Streamlit app provides three main tabs:

**1. Image Analysis** - Upload and analyze images
- Real-time prediction
- Confidence scores
- Risk assessment
- XAI explanations
- Downloadable JSON reports

**2. Model Statistics** - Performance dashboard
- Training curves (loss & accuracy)
- Confusion matrix
- Model architecture details
- Dataset information

**3. About** - System information

### What You Can Do

✅ **Upload any car damage image** - Get instant analysis
✅ **View XAI explanations** - See what the model focuses on
✅ **Check model performance** - View training statistics
✅ **Assess fraud risk** - Get automated risk ratings
✅ **Download reports** - Export analysis as JSON

---

## ⚙️ Configuration

Edit `config.yaml` to customize all settings including data splits, model architecture, training parameters, and XAI methods.

---

## 📊 Expected Performance

With 1000-1500 images:
- **Accuracy**: 80-90%
- **Training Time**: 30min - 2 hours (GPU)

---

## 📝 For More Information

See detailed documentation in the code files and inline comments.

---

## 📧 Contact

- **Repository**: [https://github.com/FeYu01/DAIA](https://github.com/FeYu01/DAIA)

---

**Built for making insurance claims more secure**