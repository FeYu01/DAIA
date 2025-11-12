"""
Test script to crop Gemini watermark from a few sample images
Run this to test before applying to all images
"""

from PIL import Image
import os

def crop_watermark_test(input_folder, output_folder, pixels_to_remove=35):
    """
    Test watermark removal on a few images
    
    Args:
        input_folder: Folder with original Gemini images
        output_folder: Folder where cropped images will be saved
        pixels_to_remove: Pixels to crop from bottom (default: 35)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Created output folder: {output_folder}")
    
    # Get list of images that start with "Gemini"
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                   and f.startswith('Gemini')]
    
    if not image_files:
        print(f"❌ No images starting with 'Gemini' found in {input_folder}")
        print(f"Available files: {all_files[:5]}")
        return
    
    # Process first 2 Gemini images as a test
    test_images = image_files[:2]
    
    print(f"\n🔍 Testing on {len(test_images)} images...")
    print(f"📏 Removing {pixels_to_remove} pixels from bottom")
    print("=" * 60)
    
    for filename in test_images:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        
        try:
            # Open image
            img = Image.open(input_path)
            width, height = img.size
            
            print(f"\n📸 Processing: {filename}")
            print(f"   Original size: {width}x{height}")
            
            # Crop bottom pixels
            cropped_img = img.crop((0, 0, width, height - pixels_to_remove))
            new_width, new_height = cropped_img.size
            
            print(f"   Cropped size:  {new_width}x{new_height}")
            print(f"   Removed:       {height - new_height} pixels from bottom")
            
            # Save cropped image
            cropped_img.save(output_path)
            print(f"   ✅ Saved to: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 TEST COMPLETE!")
    print(f"\n📂 Check your cropped images in: {output_folder}")
    print("\nNext steps:")
    print("  1. Open the cropped images and verify watermark is gone")
    print("  2. If watermark remains → increase pixels_to_remove (try 40 or 50)")
    print("  3. If too much is cut → decrease pixels_to_remove (try 25 or 30)")
    print("  4. Once satisfied, apply to all images")


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION - EDIT THESE PATHS
    # ========================================
    
    # Where your Gemini images are currently stored
    INPUT_FOLDER = "data/ai_generated"
    
    # Where to save the test cropped images
    OUTPUT_FOLDER = "test_cropped_output"
    
    # Pixels to remove from bottom (adjust as needed)
    PIXELS_TO_REMOVE = 80  # Increased to 80 to fully remove watermark
    
    # ========================================
    # RUN THE TEST
    # ========================================
    
    print("=" * 60)
    print("🧪 WATERMARK REMOVAL TEST SCRIPT")
    print("=" * 60)
    print(f"\n📁 Input folder:  {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📏 Pixels to remove: {PIXELS_TO_REMOVE}")
    
    crop_watermark_test(INPUT_FOLDER, OUTPUT_FOLDER, PIXELS_TO_REMOVE)
    
    print("\n⚠️  REMINDER: You don't actually need to remove the watermark!")
    print("   The model will resize all images to 224x224 anyway.")
    print("   Quality differences actually help the model learn.\n")
