# Quick Start Guide - DAIA

This is a step-by-step guide to get your AI detection system up and running in one week.

## 📅 Week Schedule

### Weekend (Saturday-Sunday): Data Collection
**Goal**: Collect 1000-1500 images

Follow the prompts provided earlier to generate AI images using:
- Bing Image Creator (unlimited free)
- Leonardo.AI (150 credits/day)
- Perplexity Pro (your access)
- Google Gemini Pro (your access)

Download real car damage images from:
- Kaggle datasets
- Google Images (Creative Commons)
- GitHub repositories

### Monday: Setup & Test
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run setup script
python setup.py

# 3. Verify data is loaded
python src/data_loader.py
```

### Tuesday-Wednesday: Training
```bash
# Start training (will take 1-3 hours)
python src/train.py

# Monitor progress
tail -f outputs/logs/training.log
```

**What to expect**:
- Training will run for 20 epochs (or early stop)
- Best model saved automatically
- Training curves generated
- Confusion matrix created

### Thursday: Testing & Predictions
```bash
# Test single image
python src/predict.py path/to/test/image.jpg --save-explanation

# Test batch of images
python src/predict.py data/test_images/ --batch --save-explanation
```

### Friday: Demo & Presentation
```bash
# Launch interactive demo
python src/demo.py
```

Open browser to http://localhost:7860

---

## 🎯 Minimal Working Example

If you're short on time, here's the absolute minimum:

### 1. Get 500 Images (2-3 hours)
```
- 250 real: Download from Kaggle "car damage dataset"
- 250 AI: Use Bing Image Creator with provided prompts
```

### 2. One-Command Training (1 hour)
```bash
python src/train.py
```

### 3. One-Command Demo (1 minute)
```bash
python src/demo.py
```

**Done!** You have a working prototype.

---

## 🔧 Customization Tips

### If Training is Too Slow
Edit `config.yaml`:
```yaml
data:
  batch_size: 8  # Reduce from 16

model:
  freeze_backbone: true  # Freeze ViT initially
  freeze_epochs: 5       # Train only classifier for 5 epochs
```

### If Accuracy is Low
1. **Check data balance**: Should be 50/50 real/AI
2. **Increase epochs**: Set to 30-40
3. **Lower learning rate**: Try 0.00005
4. **Add more data**: Target 1500+ images

### If Out of Memory
```yaml
data:
  batch_size: 4
  num_workers: 0
```

---

## 📊 Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Data collection | 4-6 hours | 1000-1500 images |
| Setup & testing | 1 hour | Environment ready |
| Training | 1-3 hours | Trained model |
| Testing | 30 min | Evaluation metrics |
| Demo creation | 15 min | Working UI |
| **Total** | **7-11 hours** | **Complete system** |

---

## ✅ Checklist

Before presenting your project, verify:

- [ ] Dataset organized in `data/real/` and `data/ai_generated/`
- [ ] Model trained with >80% validation accuracy
- [ ] Training curves show no overfitting
- [ ] Confusion matrix looks reasonable
- [ ] Can predict on new images successfully
- [ ] XAI explanations are generated
- [ ] Demo runs without errors
- [ ] README.md is updated with your results

---

## 🎓 For Your Presentation

### What to Show

1. **Problem Statement**
   - Insurance fraud using AI-generated images
   - Need for automated detection

2. **Your Solution**
   - Vision Transformer approach
   - Transfer learning on car damage images
   - Explainable AI for trust

3. **Demo**
   - Live prediction on test images
   - Show explanations (heatmaps)
   - Discuss confidence scores

4. **Results**
   - Training curves
   - Confusion matrix
   - Test accuracy: ~85-90%
   - Example predictions (good and bad)

5. **Limitations & Future Work**
   - Dataset size constraints
   - Specific to certain AI generators
   - Could improve with more data
   - Real-world deployment considerations

### Key Points to Emphasize

✅ **XAI is crucial** - Insurance decisions need explanations
✅ **Transfer learning** - Leveraged pre-trained ViT (don't train from scratch)
✅ **Domain-specific** - Focused on car damage, not general images
✅ **Practical constraints** - Worked with limited compute and time
✅ **Production considerations** - Confidence thresholds, manual review flags

---

## 🆘 Common Issues

### "No images found in data/"
- Make sure images are in `data/real/` and `data/ai_generated/`
- Check file extensions (.jpg, .jpeg, .png)

### "CUDA out of memory"
- Reduce batch_size in config.yaml
- Use CPU instead: Edit config and set `device: "cpu"`

### "Model accuracy stuck at 50%"
- Check data balance (should be equal)
- Verify images are actually different (real vs AI)
- Try training longer

### "Import errors"
- Run: `pip install -r requirements.txt`
- Check Python version (need 3.8+)

---

## 📞 Need Help?

1. Check the main README.md
2. Look at code comments in src/ files
3. Run test scripts (data_loader.py, model.py, explainer.py)
4. Open an issue on GitHub

---

**Good luck with your project! 🚀**
