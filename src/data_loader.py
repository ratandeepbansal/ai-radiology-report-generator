"""
Data Loader Module for MedAssist Copilot
Handles image loading, preprocessing, and data management for chest X-rays
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

import numpy as np
from PIL import Image, ImageFile
import torch
from torchvision import transforms

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XRayDataLoader:
    """
    Data loader for chest X-ray images with preprocessing capabilities
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        image_size: Tuple[int, int] = (384, 384),
        normalize: bool = True
    ):
        """
        Initialize the data loader

        Args:
            data_dir: Directory containing X-ray images
            image_size: Target size for images (height, width)
            normalize: Whether to normalize images
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize = normalize

        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']

        # Create preprocessing transforms
        self.transform = self._create_transforms()

        # Statistics
        self.stats = {
            'total_images': 0,
            'corrupted_images': 0,
            'processed_images': 0
        }

    def _create_transforms(self) -> transforms.Compose:
        """
        Create image preprocessing transforms

        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]

        if self.normalize:
            # ImageNet normalization (commonly used for pretrained models)
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return transforms.Compose(transform_list)

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load an image from disk with error handling

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            img = Image.open(image_path)

            # Convert to RGB if grayscale or RGBA
            if img.mode != 'RGB':
                img = img.convert('RGB')

            logger.info(f"Successfully loaded image: {image_path}")
            return img

        except (IOError, OSError) as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            self.stats['corrupted_images'] += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {image_path}: {str(e)}")
            self.stats['corrupted_images'] += 1
            return None

    def preprocess_image(
        self,
        image: Image.Image,
        return_type: str = 'tensor'
    ) -> Optional[Any]:
        """
        Preprocess an image for model input

        Args:
            image: PIL Image object
            return_type: 'tensor' or 'pil' or 'numpy'

        Returns:
            Preprocessed image in requested format
        """
        try:
            if return_type == 'tensor':
                # Apply transforms and return tensor
                tensor = self.transform(image)
                self.stats['processed_images'] += 1
                return tensor

            elif return_type == 'pil':
                # Resize and return PIL Image
                resized = image.resize(self.image_size, Image.Resampling.LANCZOS)
                self.stats['processed_images'] += 1
                return resized

            elif return_type == 'numpy':
                # Convert to numpy array
                resized = image.resize(self.image_size, Image.Resampling.LANCZOS)
                array = np.array(resized)
                self.stats['processed_images'] += 1
                return array

            else:
                logger.error(f"Unknown return_type: {return_type}")
                return None

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def load_and_preprocess(
        self,
        image_path: str,
        return_type: str = 'tensor'
    ) -> Optional[Any]:
        """
        Load and preprocess an image in one step

        Args:
            image_path: Path to the image file
            return_type: 'tensor' or 'pil' or 'numpy'

        Returns:
            Preprocessed image or None if loading fails
        """
        image = self.load_image(image_path)
        if image is None:
            return None

        return self.preprocess_image(image, return_type)

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get information about an image file

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image information
        """
        try:
            img = Image.open(image_path)
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB

            info = {
                'path': str(image_path),
                'filename': Path(image_path).name,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size_mb': round(file_size, 2),
                'is_valid': True
            }

            return info

        except Exception as e:
            return {
                'path': str(image_path),
                'filename': Path(image_path).name,
                'error': str(e),
                'is_valid': False
            }

    def find_images(self, directory: Optional[str] = None) -> List[str]:
        """
        Find all image files in a directory

        Args:
            directory: Directory to search (defaults to self.data_dir)

        Returns:
            List of image file paths
        """
        search_dir = Path(directory) if directory else self.data_dir

        if not search_dir.exists():
            logger.warning(f"Directory does not exist: {search_dir}")
            return []

        image_files = []
        for ext in self.supported_formats:
            image_files.extend(list(search_dir.glob(f"*{ext}")))
            image_files.extend(list(search_dir.glob(f"*{ext.upper()}")))

        self.stats['total_images'] = len(image_files)
        logger.info(f"Found {len(image_files)} images in {search_dir}")

        return [str(f) for f in sorted(image_files)]

    def get_dataset_statistics(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate statistics about the dataset

        Args:
            directory: Directory to analyze (defaults to self.data_dir)

        Returns:
            Dictionary with dataset statistics
        """
        image_files = self.find_images(directory)

        stats = {
            'total_images': len(image_files),
            'formats': {},
            'sizes': [],
            'corrupted_images': 0,
            'total_size_mb': 0
        }

        for img_path in image_files:
            info = self.get_image_info(img_path)

            if info['is_valid']:
                # Count formats
                fmt = info.get('format', 'unknown')
                stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1

                # Collect sizes
                stats['sizes'].append(info['size'])
                stats['total_size_mb'] += info['file_size_mb']
            else:
                stats['corrupted_images'] += 1

        # Calculate average size
        if stats['sizes']:
            widths = [s[0] for s in stats['sizes']]
            heights = [s[1] for s in stats['sizes']]
            stats['avg_width'] = int(np.mean(widths))
            stats['avg_height'] = int(np.mean(heights))
            stats['min_size'] = (min(widths), min(heights))
            stats['max_size'] = (max(widths), max(heights))

        stats['total_size_mb'] = round(stats['total_size_mb'], 2)

        return stats

    def create_image_batch(
        self,
        image_paths: List[str],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        """
        Create batches of preprocessed images

        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch

        Returns:
            List of batched tensors
        """
        batches = []
        current_batch = []

        for img_path in image_paths:
            tensor = self.load_and_preprocess(img_path, return_type='tensor')

            if tensor is not None:
                current_batch.append(tensor)

                if len(current_batch) == batch_size:
                    batches.append(torch.stack(current_batch))
                    current_batch = []

        # Add remaining images as final batch
        if current_batch:
            batches.append(torch.stack(current_batch))

        logger.info(f"Created {len(batches)} batches from {len(image_paths)} images")
        return batches

    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate if an image can be loaded and processed

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_valid, message)
        """
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "File does not exist"

        # Check file extension
        ext = Path(image_path).suffix.lower()
        if ext not in self.supported_formats:
            return False, f"Unsupported format: {ext}"

        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            return False, f"File too large: {file_size_mb:.2f}MB"

        # Try to load the image
        img = self.load_image(image_path)
        if img is None:
            return False, "Failed to load image"

        # Try to preprocess
        processed = self.preprocess_image(img, return_type='tensor')
        if processed is None:
            return False, "Failed to preprocess image"

        return True, "Image is valid"

    def get_stats(self) -> Dict[str, int]:
        """
        Get current loader statistics

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_images': 0,
            'corrupted_images': 0,
            'processed_images': 0
        }


# Utility functions

def display_image_grid(
    images: List[Image.Image],
    titles: Optional[List[str]] = None,
    rows: int = 2,
    cols: int = 3,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Display a grid of images (useful for Jupyter notebooks)

    Args:
        images: List of PIL Images
        titles: Optional list of titles for each image
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
            ax.axis('off')

            if titles and idx < len(titles):
                ax.set_title(titles[idx])

        # Hide unused subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning("matplotlib not installed. Cannot display image grid.")
    except Exception as e:
        logger.error(f"Error displaying images: {str(e)}")


def save_preprocessed_images(
    image_paths: List[str],
    output_dir: str,
    loader: XRayDataLoader
):
    """
    Preprocess and save images to a directory

    Args:
        image_paths: List of input image paths
        output_dir: Directory to save preprocessed images
        loader: XRayDataLoader instance
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img = loader.load_and_preprocess(img_path, return_type='pil')

        if img is not None:
            filename = Path(img_path).name
            output_file = output_path / filename
            img.save(output_file)
            logger.info(f"Saved preprocessed image: {output_file}")


# Main execution for testing
if __name__ == "__main__":
    print("=" * 60)
    print("MedAssist Copilot - Data Loader Module Test")
    print("=" * 60)

    # Initialize loader
    loader = XRayDataLoader(
        data_dir="data/raw",
        image_size=(384, 384),
        normalize=True
    )

    print("\n✅ Data loader initialized")
    print(f"   - Data directory: {loader.data_dir}")
    print(f"   - Image size: {loader.image_size}")
    print(f"   - Normalize: {loader.normalize}")

    # Find images
    print("\n🔍 Searching for images...")
    images = loader.find_images()

    if images:
        print(f"   Found {len(images)} images")

        # Get dataset statistics
        print("\n📊 Dataset Statistics:")
        stats = loader.get_dataset_statistics()
        print(f"   - Total images: {stats['total_images']}")
        print(f"   - Formats: {stats['formats']}")
        print(f"   - Total size: {stats['total_size_mb']} MB")

        if stats.get('avg_width'):
            print(f"   - Average size: {stats['avg_width']}x{stats['avg_height']}")
            print(f"   - Min size: {stats['min_size']}")
            print(f"   - Max size: {stats['max_size']}")

        # Test loading first image
        print("\n🖼️  Testing image loading...")
        test_image = images[0]
        img = loader.load_image(test_image)

        if img:
            print(f"   ✅ Successfully loaded: {Path(test_image).name}")

            # Test preprocessing
            print("\n⚙️  Testing preprocessing...")
            tensor = loader.preprocess_image(img, return_type='tensor')

            if tensor is not None:
                print(f"   ✅ Preprocessed shape: {tensor.shape}")
            else:
                print("   ❌ Preprocessing failed")
        else:
            print("   ❌ Failed to load image")
    else:
        print("   ℹ️  No images found in data/raw/")
        print("   Add some test images to data/raw/ to test the loader")

    # Show statistics
    print("\n📈 Loader Statistics:")
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
