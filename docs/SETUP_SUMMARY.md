# 📦 Installation Complete!

## ✅ What Was Installed

**Total: 20+ packages (~3.4 GB)**

### Core Packages Installed:
- **PyTorch 2.9.0** - Deep learning framework
- **Transformers 4.57.1** - HuggingFace models (ViT)
- **Streamlit 1.51.0** - Web dashboard
- **Gradio 5.49.1** - Simple demo interface
- **Albumentations 2.0.8** - Image augmentation
- **Grad-CAM 1.5.5** - XAI explanations
- **And 100+ dependencies...**

---

## 🔧 Installation Method Used

✅ **VIRTUAL ENVIRONMENT** (Recommended)

```bash
# Location of your virtual environment
/workspaces/DAIA/venv/
```

### How to Activate in Future Sessions

**Every time you open a new terminal:**

```bash
cd /workspaces/DAIA
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when activated.

---

## 🚀 Next Steps

### 1. Verify Installation
```bash
source venv/bin/activate
python setup.py
```

### 2. Start Collecting Data
Use the 100 prompts provided earlier to generate images with:
- Bing Image Creator (unlimited)
- Leonardo.AI (150 credits/day)
- Perplexity Pro
- Google Gemini Pro

**Target: 1000-1500 images by Sunday**

### 3. Organize Your Data
```
data/
├── real/              # Put 500-750 real car damage images here
└── ai_generated/      # Put 500-750 AI-generated images here
```

### 4. Train the Model (Monday-Wednesday)
```bash
source venv/bin/activate
python src/train.py
```

### 5. Launch Demo (Thursday-Friday)
```bash
source venv/bin/activate
streamlit run src/app.py
```

---

## 📚 Quick Command Reference

```bash
# Activate environment
source venv/bin/activate

# Train model
python src/train.py

# Single prediction
python src/predict.py path/to/image.jpg

# Launch Streamlit dashboard
streamlit run src/app.py

# Launch Gradio demo
python src/demo.py

# Deactivate environment (when done)
deactivate
```

---

## 💡 Important Notes

### ⚠️ Always Activate Environment First!
Before running ANY command, activate the virtual environment:
```bash
source venv/bin/activate
```

### 📁 File Structure
```
DAIA/
├── venv/                    # Virtual environment (DON'T commit to git)
├── data/                    # Your dataset
├── src/                     # All Python code
├── outputs/                 # Training results (created automatically)
├── config.yaml              # Configuration
└── requirements.txt         # Package list
```

### 🔄 If You Need to Reinstall
```bash
# Delete virtual environment
rm -rf venv

# Recreate and reinstall
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ❓ Troubleshooting

### "Command not found: python"
Try: `python3 -m venv venv` and `python3 src/train.py`

### "ModuleNotFoundError"
Make sure you activated the environment:
```bash
source venv/bin/activate
```

### "Permission denied"
The dev container should have permissions. If not:
```bash
chmod +x src/*.py
```

### Streamlit won't start
```bash
# Make sure you're in project root
cd /workspaces/DAIA
source venv/bin/activate
streamlit run src/app.py
```

---

## 📊 Project Timeline (1 Week)

**Weekend (Sat-Sun):**
- ✅ Environment setup (DONE!)
- 🔄 Data collection (500-750 each class)

**Monday-Wednesday:**
- Train model (~1-3 hours)
- Validate results

**Thursday-Friday:**
- Test demo interface
- Prepare presentation
- Practice 5-minute demo

---

## 🎯 You're All Set!

Your environment is ready. All packages installed in an isolated virtual environment.

**Next action:** Start collecting images this weekend using the prompts!

Remember: `source venv/bin/activate` before every session! 🚀
