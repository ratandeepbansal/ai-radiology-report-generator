"""
View Sample Images from Kaggle Dataset
Displays sample NORMAL and PNEUMONIA X-ray images
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import XRayDataLoader


def view_samples():
    """Display sample images from both categories"""
    print("\n" + "=" * 70)
    print(" " * 20 + "SAMPLE IMAGE VIEWER")
    print("=" * 70)

    data_dir = Path("data/raw")

    # Get image lists
    normal_images = list((data_dir / "NORMAL").glob("*.jpeg"))
    pneumonia_images = list((data_dir / "PNEUMONIA").glob("*.jpeg"))

    print(f"\n📁 Dataset contains:")
    print(f"   • NORMAL: {len(normal_images)} images")
    print(f"   • PNEUMONIA: {len(pneumonia_images)} images")

    # Initialize loader
    loader = XRayDataLoader()

    # Select random samples
    num_samples = 3
    normal_samples = random.sample(normal_images, min(num_samples, len(normal_images)))
    pneumonia_samples = random.sample(pneumonia_images, min(num_samples, len(pneumonia_images)))

    print(f"\n🔍 Viewing {num_samples} random samples from each category...")

    # Display NORMAL samples
    print("\n" + "=" * 70)
    print("  NORMAL X-RAYS")
    print("=" * 70)

    for i, img_path in enumerate(normal_samples, 1):
        print(f"\n{i}. {img_path.name}")

        # Get image info
        info = loader.get_image_info(str(img_path))
        if info['is_valid']:
            print(f"   Size: {info['width']}x{info['height']} pixels")
            print(f"   File size: {info['file_size_mb']} MB")
            print(f"   Format: {info['format']}")

        # Load and get details
        img = loader.load_image(str(img_path))
        if img:
            tensor = loader.preprocess_image(img, return_type='tensor')
            print(f"   Preprocessed: {tensor.shape} ✅")
        else:
            print(f"   ❌ Failed to load")

    # Display PNEUMONIA samples
    print("\n" + "=" * 70)
    print("  PNEUMONIA X-RAYS")
    print("=" * 70)

    for i, img_path in enumerate(pneumonia_samples, 1):
        print(f"\n{i}. {img_path.name}")

        # Determine type
        if 'bacteria' in img_path.name.lower():
            ptype = "Bacterial"
        elif 'virus' in img_path.name.lower():
            ptype = "Viral"
        else:
            ptype = "Unknown"
        print(f"   Type: {ptype}")

        # Get image info
        info = loader.get_image_info(str(img_path))
        if info['is_valid']:
            print(f"   Size: {info['width']}x{info['height']} pixels")
            print(f"   File size: {info['file_size_mb']} MB")
            print(f"   Format: {info['format']}")

        # Load and get details
        img = loader.load_image(str(img_path))
        if img:
            tensor = loader.preprocess_image(img, return_type='tensor')
            print(f"   Preprocessed: {tensor.shape} ✅")
        else:
            print(f"   ❌ Failed to load")

    print("\n" + "=" * 70)
    print("\n✅ All sample images loaded successfully!")
    print("\n💡 To view images visually, you can:")
    print("   1. Open the files directly in data/raw/NORMAL or data/raw/PNEUMONIA")
    print("   2. Use a Jupyter notebook with matplotlib")
    print("   3. Use the Streamlit app (coming in Week 4)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        view_samples()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
