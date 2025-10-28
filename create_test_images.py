"""
Utility script to create sample test images for development
Generates synthetic chest X-ray-like images for testing the pipeline
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


def create_synthetic_xray(
    width: int = 512,
    height: int = 512,
    label: str = "Normal"
) -> Image.Image:
    """
    Create a synthetic chest X-ray-like image for testing

    Args:
        width: Image width
        height: Image height
        label: Label to add to the image

    Returns:
        PIL Image object
    """
    # Create a grayscale base image
    # Use varying gray tones to simulate X-ray appearance
    base_color = random.randint(40, 80)
    img_array = np.random.randint(
        base_color - 20,
        base_color + 20,
        (height, width),
        dtype=np.uint8
    )

    # Add some structure to simulate anatomical features
    # Create a lighter central region (lung fields)
    center_y, center_x = height // 2, width // 2
    for i in range(height):
        for j in range(width):
            # Distance from center
            dist_from_center = np.sqrt((i - center_y)**2 + (j - center_x)**2)

            # Create lighter regions in the center
            if dist_from_center < min(width, height) * 0.35:
                img_array[i, j] = min(255, img_array[i, j] + 40)

    # Add some darker areas to simulate ribs/bones
    num_ribs = 6
    for rib in range(num_ribs):
        y_pos = int(height * 0.3 + (rib * height * 0.08))
        for j in range(width):
            # Create curved rib-like structures
            curve = int(20 * np.sin(j / width * np.pi))
            y = y_pos + curve
            if 0 <= y < height:
                # Make a band darker
                for offset in range(-3, 4):
                    if 0 <= y + offset < height:
                        img_array[y + offset, j] = max(0, img_array[y + offset, j] - 30)

    # Convert to PIL Image
    img = Image.fromarray(img_array, mode='L')

    # Convert to RGB
    img = img.convert('RGB')

    # Add label
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None

    # Add label in corner
    text_position = (10, 10)
    draw.text(text_position, label, fill=(255, 255, 255), font=font)

    # Add timestamp/marker
    marker = f"TEST_{random.randint(1000, 9999)}"
    marker_position = (width - 100, height - 30)
    draw.text(marker_position, marker, fill=(200, 200, 200), font=font)

    return img


def generate_test_images(output_dir: str = "data/raw", num_images: int = 5):
    """
    Generate multiple test images with different labels

    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = [
        "Normal",
        "Pneumonia",
        "Effusion",
        "Cardiomegaly",
        "Nodule"
    ]

    print("🖼️  Generating synthetic test images...")
    print("=" * 60)

    for i in range(num_images):
        label = labels[i % len(labels)]
        filename = f"test_xray_{i+1:02d}_{label.lower()}.jpg"
        filepath = output_path / filename

        # Create image with some variation
        img = create_synthetic_xray(
            width=512,
            height=512,
            label=label
        )

        # Save image
        img.save(filepath, quality=95)
        print(f"✅ Created: {filename}")

    print("=" * 60)
    print(f"✨ Successfully generated {num_images} test images in {output_dir}")
    print("\nNote: These are synthetic images for testing only.")
    print("Download real chest X-ray datasets from:")
    print("  - Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("  - MIMIC-CXR: https://physionet.org/content/mimic-cxr/")


def create_sample_comparison_images(output_dir: str = "data/raw"):
    """
    Create a pair of images for testing comparison features

    Args:
        output_dir: Directory to save images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create "old" image (normal)
    old_img = create_synthetic_xray(width=512, height=512, label="Previous - Normal")
    old_img.save(output_path / "comparison_old.jpg", quality=95)

    # Create "new" image (with finding)
    new_img = create_synthetic_xray(width=512, height=512, label="Current - Infiltrate")

    # Add a dark spot to simulate new finding
    draw = ImageDraw.Draw(new_img)
    draw.ellipse([(300, 200), (350, 250)], fill=(80, 80, 80))

    new_img.save(output_path / "comparison_new.jpg", quality=95)

    print("\n✅ Created comparison image pair:")
    print("   - comparison_old.jpg")
    print("   - comparison_new.jpg")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MedAssist Copilot - Test Image Generator")
    print("=" * 60 + "\n")

    # Check if data/raw directory has images
    data_dir = Path("data/raw")
    existing_images = []

    if data_dir.exists():
        existing_images = [
            f for f in data_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]

    if existing_images:
        print(f"ℹ️  Found {len(existing_images)} existing images in data/raw/")
        response = input("Generate additional test images? (y/n): ")
        if response.lower() != 'y':
            print("Skipping image generation.")
            exit(0)

    # Generate test images
    try:
        num_images = int(input("\nHow many test images to generate? (default: 5): ") or "5")
    except ValueError:
        num_images = 5

    generate_test_images(num_images=num_images)

    # Ask about comparison images
    response = input("\nGenerate comparison images for testing? (y/n): ")
    if response.lower() == 'y':
        create_sample_comparison_images()

    print("\n✨ Done! You can now test the data pipeline.")
    print("Run: python src/data_loader.py")
    print("=" * 60 + "\n")
