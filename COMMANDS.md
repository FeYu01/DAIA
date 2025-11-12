# DAIA - Quick Reference

**Always activate environment first:** `source venv/bin/activate`

---

## 📊 Check Dataset Status
```bash
python scripts/setup.py
```

---

## 🔪 Remove Gemini Watermarks
```bash
python scripts/crop_gemini_only.py
```

---

## 🎓 Train Model
```bash
python src/train.py
```

---

## 🔮 Make Predictions
```bash
# Single image
python src/predict.py path/to/image.jpg

# With explanation
python src/predict.py path/to/image.jpg --save-explanation
```

---

## 🌐 Launch Demos

**Streamlit (Full Dashboard):**
```bash
streamlit run src/app.py
```

**Gradio (Simple):**
```bash
python src/demo.py
```

---

## 📁 Project Structure

```
DAIA/
├── src/           # Main code
├── scripts/       # Helper scripts
├── docs/          # Documentation
├── data/          # Dataset
└── outputs/       # Generated files
```

---

## 📖 Full Documentation

See [docs/](docs/) folder for detailed guides.

---

## 🆘 Quick Help

Data location: The project expects data at the path configured in `config.yaml`. To avoid committing large files into the repository, datasets are stored outside the repository at:

```
/home/codespace/datasets/DAIA/
	├─ real/
	└─ ai_generated/
```

To inspect the dataset (or copy it into the workspace for temporary use):

```bash
# Check dataset
bash scripts/fetch_data.sh

# Copy files into workspace (optional):
bash scripts/fetch_data.sh --copy-to-workspace
```

**Problem:** Module not found  
**Solution:** Activate environment: `source venv/bin/activate`

**Problem:** No model found  
**Solution:** Train first: `python src/train.py`

**Problem:** Gemini watermark visible  
**Solution:** Run: `python scripts/crop_gemini_only.py`

For detailed help, see [docs/](docs/)
