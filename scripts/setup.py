"""
Setup script for DAIA project
Creates necessary directories and checks dependencies
"""

import os
import sys


def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        "data/real",
        "data/ai_generated",
        "models/checkpoints",
        "outputs/explanations",
        "outputs/plots",
        "outputs/logs",
        "notebooks"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\nDirectory structure created successfully!")


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("sklearn", "scikit-learn"),
        ("albumentations", "Albumentations"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("numpy", "NumPy"),
        ("gradio", "Gradio")
    ]
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_dataset():
    """Check if dataset is present"""
    print("\nChecking dataset...")
    
    real_dir = "data/real"
    ai_dir = "data/ai_generated"
    
    real_count = len([f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(real_dir) else 0
    ai_count = len([f for f in os.listdir(ai_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(ai_dir) else 0
    
    if real_count == 0 and ai_count == 0:
        print("  ⚠️  No images found!")
        print("\n📝 To get started:")
        print("  1. Place real car damage images in: data/real/")
        print("  2. Place AI-generated images in: data/ai_generated/")
        print("  3. Aim for 500-1000 images per class")
        return False
    else:
        print(f"  ✓ Real images: {real_count}")
        print(f"  ✓ AI-generated images: {ai_count}")
        print(f"  ✓ Total: {real_count + ai_count}")
        
        if real_count + ai_count < 200:
            print("\n  ⚠️  Dataset is small (< 200 images)")
            print("     Recommended: 1000-1500 total images for good performance")
        
        return True


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:\n")
    
    print("1. Prepare your dataset:")
    print("   - Add images to data/real/ and data/ai_generated/")
    print("   - Recommended: 1000-1500 total images (500-750 each)")
    
    print("\n2. (Optional) Customize configuration:")
    print("   - Edit config.yaml to adjust hyperparameters")
    
    print("\n3. Train the model:")
    print("   python src/train.py")
    
    print("\n4. Make predictions:")
    print("   python src/predict.py path/to/image.jpg --save-explanation")
    
    print("\n5. Run web demo:")
    print("   python src/demo.py")
    
    print("\n" + "="*60)
    print("\n📚 For more information, see README.md")
    print("❓ Issues? Check: https://github.com/FeYu01/DAIA/issues\n")


def main():
    """Main setup function"""
    print("="*60)
    print("DAIA - Setup Script")
    print("="*60)
    print()
    
    # Create directories
    create_directory_structure()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check dataset
    data_ok = check_dataset()
    
    # Print next steps
    print_next_steps()
    
    # Summary
    if deps_ok and data_ok:
        print("✅ Setup successful! You're ready to train.")
    elif deps_ok:
        print("⚠️  Please add dataset before training.")
    else:
        print("⚠️  Please install dependencies first.")


if __name__ == "__main__":
    main()
