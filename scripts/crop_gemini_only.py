"""
Crop ONLY Gemini images (files starting with 'Gemini')
Leaves Leonardo.AI and other images untouched
"""

from PIL import Image
import os
import shutil

def crop_gemini_images(input_folder, pixels_to_remove=80):
    """
    Crop only images starting with 'Gemini' in filename
    Replaces original files with cropped versions
    """
    # Get all Gemini images
    all_files = os.listdir(input_folder)
    gemini_files = [f for f in all_files 
                    if f.startswith('Gemini') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not gemini_files:
        print("❌ No Gemini images found!")
        return
    
    print(f"🔍 Found {len(gemini_files)} Gemini images")
    print(f"📏 Will remove {pixels_to_remove} pixels from bottom")
    print("=" * 60)
    
    # Backup original files first
    backup_folder = os.path.join(input_folder, "gemini_originals_backup")
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"✅ Created backup folder: {backup_folder}")
    
    success = 0
    
    for filename in gemini_files:
        input_path = os.path.join(input_folder, filename)
        backup_path = os.path.join(backup_folder, filename)
        
        try:
            # Backup original
            shutil.copy2(input_path, backup_path)
            
            # Open and crop
            img = Image.open(input_path)
            width, height = img.size
            
            cropped_img = img.crop((0, 0, width, height - pixels_to_remove))
            
            # Overwrite original with cropped version
            cropped_img.save(input_path)
            
            success += 1
            print(f"✅ {filename}: {width}x{height} → {width}x{height-pixels_to_remove}")
            print(f"   Original backed up to: gemini_originals_backup/")
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Successfully cropped {success}/{len(gemini_files)} images!")
    print(f"💾 Originals saved in: {backup_folder}")
    print(f"✅ Cropped images replaced originals in: {input_folder}")


if __name__ == "__main__":
    INPUT_FOLDER = "data/ai_generated"
    PIXELS_TO_REMOVE = 80  # Tested and confirmed
    
    print("=" * 60)
    print("🔪 CROP GEMINI WATERMARKS ONLY")
    print("=" * 60)
    print(f"\n📁 Folder: {INPUT_FOLDER}")
    print(f"🎯 Target: Images starting with 'Gemini'")
    print(f"📏 Crop: {PIXELS_TO_REMOVE} pixels from bottom")
    print(f"💾 Backup: Will save originals to gemini_originals_backup/")
    
    response = input("\nProceed? (yes/no): ").lower().strip()
    
    if response == 'yes':
        crop_gemini_images(INPUT_FOLDER, PIXELS_TO_REMOVE)
        print("\n✅ Done! Your Gemini images are now watermark-free!")
        print("Leonardo.AI images were left untouched.")
    else:
        print("\n❌ Cancelled. No changes made.")
