"""
Full script to crop Gemini watermark from ALL images
⚠️  Only run this AFTER testing with test_watermark_crop.py
"""

from PIL import Image
import os

def crop_all_images(input_folder, output_folder, pixels_to_remove=35, add_prefix=True):
    """
    Crop watermark from all images in folder
    
    Args:
        input_folder: Folder with original Gemini images
        output_folder: Folder where cropped images will be saved
        pixels_to_remove: Pixels to crop from bottom
        add_prefix: Add 'cropped_' prefix to filename
    """
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ Created output folder: {output_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in {input_folder}")
        return
    
    total = len(image_files)
    success = 0
    failed = 0
    
    print(f"\n🚀 Processing {total} images...")
    print(f"📏 Removing {pixels_to_remove} pixels from bottom")
    print("=" * 60)
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        
        # Output filename
        if add_prefix:
            output_filename = f"cropped_{filename}"
        else:
            output_filename = filename
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Open and crop
            img = Image.open(input_path)
            width, height = img.size
            
            cropped_img = img.crop((0, 0, width, height - pixels_to_remove))
            cropped_img.save(output_path)
            
            success += 1
            print(f"✅ [{i}/{total}] {filename} → {output_filename}")
            
        except Exception as e:
            failed += 1
            print(f"❌ [{i}/{total}] Error with {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print(f"✅ Success: {success}/{total}")
    if failed > 0:
        print(f"❌ Failed:  {failed}/{total}")
    print(f"\n📂 Cropped images saved to: {output_folder}")


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Input: Current Gemini images location
    INPUT_FOLDER = "data/ai_generated"
    
    # Output: Where to save cropped images
    # Option 1: Save to new folder, keep originals
    OUTPUT_FOLDER = "data/ai_generated_cropped"
    
    # Option 2: Overwrite originals (BE CAREFUL!)
    # OUTPUT_FOLDER = "data/ai_generated"  # Uncomment to overwrite
    
    # Pixels to remove (adjust based on your test)
    PIXELS_TO_REMOVE = 80  # Tested and confirmed to remove Gemini watermark
    
    # Add 'cropped_' prefix to filenames?
    ADD_PREFIX = True  # Set to False to keep original names
    
    # ========================================
    # SAFETY CHECK
    # ========================================
    
    print("=" * 60)
    print("⚠️  FULL BATCH WATERMARK REMOVAL")
    print("=" * 60)
    print(f"\n📁 Input:  {INPUT_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print(f"📏 Pixels to remove: {PIXELS_TO_REMOVE}")
    print(f"🏷️  Add prefix: {ADD_PREFIX}")
    
    # Count images
    image_count = len([f for f in os.listdir(INPUT_FOLDER) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📊 Found {image_count} images to process")
    
    # Confirm before processing
    print("\n⚠️  WARNING: Make sure you tested with test_watermark_crop.py first!")
    response = input("\nProceed with batch processing? (yes/no): ").lower().strip()
    
    if response == 'yes':
        crop_all_images(INPUT_FOLDER, OUTPUT_FOLDER, PIXELS_TO_REMOVE, ADD_PREFIX)
    else:
        print("\n❌ Cancelled. No changes made.")
        print("💡 Run test_watermark_crop.py first to verify settings!")
