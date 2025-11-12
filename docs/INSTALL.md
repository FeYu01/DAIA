# Installation Guide - DAIA

Complete installation instructions for the DAIA project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM recommended
- GPU (optional, but recommended for faster training)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FeYu01/DAIA.git
cd DAIA
```

### 2. Create Virtual Environment (Recommended)

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will install ~2-3GB of packages. Installation may take 5-10 minutes depending on your internet connection.

### 4. Run Setup Script

```bash
python setup.py
```

This will:
- Create necessary directories
- Check if all dependencies are installed
- Verify your dataset (if present)
- Show next steps

### 5. Verify Installation

Test that everything is installed correctly:

```bash
# Test utilities
python -c "import sys; sys.path.insert(0, 'src'); from utils import load_config; print('✓ Utils OK')"

# Test data loader (requires transformers)
python -c "import sys; sys.path.insert(0, 'src'); from data_loader import CarDamageDataset; print('✓ Data loader OK')"

# Test model (requires transformers)
python -c "import sys; sys.path.insert(0, 'src'); from model import ViTClassifier; print('✓ Model OK')"
```

If all tests pass, you're ready to go! ✓

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch manually based on your system:

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For other configurations**, visit: https://pytorch.org/get-started/locally/

### Issue: "ImportError: cannot import name 'ViTModel'"

**Solution**: Update transformers:
```bash
pip install --upgrade transformers
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Use CPU instead (slower but works):
   - Edit `config.yaml` and comment out GPU settings
2. Reduce batch size in `config.yaml`:
   ```yaml
   data:
     batch_size: 4  # or even 2
   ```

### Issue: "No images found in data/"

**Solution**: Make sure you have images in the correct directories:
```
data/
├── real/
│   └── (your real images here)
└── ai_generated/
    └── (your AI-generated images here)
```

### Issue: Path errors when running scripts

**Solution**: Always run scripts from the project root:
```bash
# Correct ✓
cd /path/to/DAIA
python src/train.py

# Incorrect ✗
cd /path/to/DAIA/src
python train.py
```

---

## Minimal Installation (For Testing Only)

If you just want to test the code structure without training:

```bash
# Install only essential packages
pip install torch torchvision transformers pillow pyyaml numpy

# Run setup
python setup.py
```

---

## Installation on Different Systems

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Clone and install
git clone https://github.com/FeYu01/DAIA.git
cd DAIA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Clone and install
git clone https://github.com/FeYu01/DAIA.git
cd DAIA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
# Install Python from python.org
# Then in PowerShell or Command Prompt:

git clone https://github.com/FeYu01/DAIA.git
cd DAIA
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Google Colab (Free GPU)

```python
# In a Colab notebook cell:
!git clone https://github.com/FeYu01/DAIA.git
%cd DAIA
!pip install -r requirements.txt

# Upload your dataset or mount Google Drive
# Then proceed with training
```

---

## Uninstallation

To remove DAIA:

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf DAIA
```

---

## Next Steps

After installation:

1. **Prepare your dataset** - See QUICKSTART.md
2. **Train the model** - Run `python src/train.py`
3. **Make predictions** - Run `python src/predict.py <image_path>`
4. **Launch demo** - Run `python src/demo.py`

For detailed usage, see README.md and QUICKSTART.md.

---

## Getting Help

- 📖 Check README.md for usage examples
- 🚀 See QUICKSTART.md for a step-by-step guide
- 🐛 Report issues: https://github.com/FeYu01/DAIA/issues
- 📧 Contact: See README.md for details
