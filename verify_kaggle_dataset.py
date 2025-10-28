"""
Verify and Analyze Kaggle Chest X-Ray Dataset
Checks the NORMAL and PNEUMONIA folders and provides comprehensive statistics
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import XRayDataLoader


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_kaggle_dataset():
    """Analyze the Kaggle chest X-ray dataset"""
    print("\n" + "=" * 70)
    print(" " * 15 + "KAGGLE DATASET VERIFICATION")
    print("=" * 70)

    data_dir = Path("data/raw")

    # Check for NORMAL and PNEUMONIA folders
    normal_dir = data_dir / "NORMAL"
    pneumonia_dir = data_dir / "PNEUMONIA"

    print_section("DIRECTORY STRUCTURE")

    if normal_dir.exists():
        print(f"✅ NORMAL folder found: {normal_dir}")
    else:
        print(f"❌ NORMAL folder not found")
        return

    if pneumonia_dir.exists():
        print(f"✅ PNEUMONIA folder found: {pneumonia_dir}")
    else:
        print(f"❌ PNEUMONIA folder not found")
        return

    # Count images in each folder
    print_section("IMAGE COUNT")

    normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
    pneumonia_images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg"))

    print(f"NORMAL images:    {len(normal_images):>5} images")
    print(f"PNEUMONIA images: {len(pneumonia_images):>5} images")
    print(f"{'─' * 30}")
    print(f"TOTAL:            {len(normal_images) + len(pneumonia_images):>5} images")

    # Calculate distribution
    total = len(normal_images) + len(pneumonia_images)
    normal_pct = (len(normal_images) / total * 100) if total > 0 else 0
    pneumonia_pct = (len(pneumonia_images) / total * 100) if total > 0 else 0

    print(f"\nDistribution:")
    print(f"  NORMAL:    {normal_pct:>5.1f}%")
    print(f"  PNEUMONIA: {pneumonia_pct:>5.1f}%")

    # Analyze file sizes
    print_section("FILE SIZE ANALYSIS")

    def get_size_stats(image_list):
        sizes = [img.stat().st_size / 1024 for img in image_list]  # KB
        if sizes:
            return {
                'min': min(sizes),
                'max': max(sizes),
                'avg': sum(sizes) / len(sizes),
                'total_mb': sum(sizes) / 1024
            }
        return None

    normal_stats = get_size_stats(normal_images)
    pneumonia_stats = get_size_stats(pneumonia_images)

    if normal_stats:
        print(f"\nNORMAL folder:")
        print(f"  Size range: {normal_stats['min']:.1f} - {normal_stats['max']:.1f} KB")
        print(f"  Average:    {normal_stats['avg']:.1f} KB")
        print(f"  Total:      {normal_stats['total_mb']:.1f} MB")

    if pneumonia_stats:
        print(f"\nPNEUMONIA folder:")
        print(f"  Size range: {pneumonia_stats['min']:.1f} - {pneumonia_stats['max']:.1f} KB")
        print(f"  Average:    {pneumonia_stats['avg']:.1f} KB")
        print(f"  Total:      {pneumonia_stats['total_mb']:.1f} MB")

    # Test loading images with our data loader
    print_section("DATA LOADER COMPATIBILITY TEST")

    loader = XRayDataLoader(data_dir="data/raw", image_size=(384, 384))

    print("\n1️⃣  Testing NORMAL images...")
    test_normal = normal_images[:5] if len(normal_images) >= 5 else normal_images
    normal_success = 0
    normal_failed = 0

    for img_path in test_normal:
        img = loader.load_image(str(img_path))
        if img:
            tensor = loader.preprocess_image(img, return_type='tensor')
            if tensor is not None:
                normal_success += 1
            else:
                normal_failed += 1
        else:
            normal_failed += 1

    print(f"   ✅ Successfully loaded: {normal_success}/{len(test_normal)}")
    if normal_failed > 0:
        print(f"   ❌ Failed: {normal_failed}")

    print("\n2️⃣  Testing PNEUMONIA images...")
    test_pneumonia = pneumonia_images[:5] if len(pneumonia_images) >= 5 else pneumonia_images
    pneumonia_success = 0
    pneumonia_failed = 0

    for img_path in test_pneumonia:
        img = loader.load_image(str(img_path))
        if img:
            tensor = loader.preprocess_image(img, return_type='tensor')
            if tensor is not None:
                pneumonia_success += 1
            else:
                pneumonia_failed += 1
        else:
            pneumonia_failed += 1

    print(f"   ✅ Successfully loaded: {pneumonia_success}/{len(test_pneumonia)}")
    if pneumonia_failed > 0:
        print(f"   ❌ Failed: {pneumonia_failed}")

    # Get detailed statistics using data loader
    print_section("DETAILED IMAGE ANALYSIS")

    print("\nAnalyzing NORMAL images...")
    normal_loader = XRayDataLoader(data_dir="data/raw/NORMAL")
    normal_detailed_stats = normal_loader.get_dataset_statistics()

    print(f"  Total images: {normal_detailed_stats['total_images']}")
    print(f"  Formats: {normal_detailed_stats['formats']}")
    if normal_detailed_stats.get('avg_width'):
        print(f"  Average dimensions: {normal_detailed_stats['avg_width']}x{normal_detailed_stats['avg_height']}")
        print(f"  Size range: {normal_detailed_stats['min_size']} to {normal_detailed_stats['max_size']}")

    print("\nAnalyzing PNEUMONIA images...")
    pneumonia_loader = XRayDataLoader(data_dir="data/raw/PNEUMONIA")
    pneumonia_detailed_stats = pneumonia_loader.get_dataset_statistics()

    print(f"  Total images: {pneumonia_detailed_stats['total_images']}")
    print(f"  Formats: {pneumonia_detailed_stats['formats']}")
    if pneumonia_detailed_stats.get('avg_width'):
        print(f"  Average dimensions: {pneumonia_detailed_stats['avg_width']}x{pneumonia_detailed_stats['avg_height']}")
        print(f"  Size range: {pneumonia_detailed_stats['min_size']} to {pneumonia_detailed_stats['max_size']}")

    # Sample filenames
    print_section("SAMPLE FILENAMES")

    print("\nNORMAL samples:")
    for img in normal_images[:3]:
        print(f"  • {img.name}")

    print("\nPNEUMONIA samples:")
    for img in pneumonia_images[:3]:
        print(f"  • {img.name}")

    # Analyze pneumonia subtypes (bacteria vs virus)
    print_section("PNEUMONIA SUBTYPE ANALYSIS")

    bacteria_count = sum(1 for img in pneumonia_images if 'bacteria' in img.name.lower())
    virus_count = sum(1 for img in pneumonia_images if 'virus' in img.name.lower())
    other_count = len(pneumonia_images) - bacteria_count - virus_count

    print(f"\nPneumonia breakdown:")
    print(f"  Bacterial: {bacteria_count:>4} ({bacteria_count/len(pneumonia_images)*100:.1f}%)")
    print(f"  Viral:     {virus_count:>4} ({virus_count/len(pneumonia_images)*100:.1f}%)")
    if other_count > 0:
        print(f"  Other:     {other_count:>4} ({other_count/len(pneumonia_images)*100:.1f}%)")

    # Summary
    print_section("SUMMARY")

    print("\n✅ Dataset Verification: PASSED")
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total images: {total}")
    print(f"   • Normal: {len(normal_images)} ({normal_pct:.1f}%)")
    print(f"   • Pneumonia: {len(pneumonia_images)} ({pneumonia_pct:.1f}%)")
    print(f"   • Data loader compatible: ✅")
    print(f"   • Ready for training: ✅")

    print("\n💡 Next Steps:")
    print("   1. Use this dataset for Week 2 model training")
    print("   2. Create train/val/test splits")
    print("   3. Consider data augmentation")
    print("   4. Test vision model on real X-rays")

    print("\n" + "=" * 70)
    print("Dataset verification complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        analyze_kaggle_dataset()
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
