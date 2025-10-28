"""
HDF5 Image Converter
Converts JPG/PNG images to HDF5 format required by CheXzero

Author: MedAssist Copilot Team
"""

import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List
import tempfile
import os


class H5Converter:
    """Converts images to HDF5 format for CheXzero"""

    def __init__(self, image_size: tuple = (224, 224), normalize: bool = True):
        """
        Initialize H5 converter

        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.image_size = image_size
        self.normalize = normalize

    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess a single image

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image as numpy array
        """
        # Open image
        img = Image.open(image_path)

        # Convert to grayscale (chest X-rays are grayscale)
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to target size
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize if requested
        if self.normalize:
            img_array = img_array / 255.0

        return img_array

    def convert_single_image(
        self,
        image_path: Union[str, Path],
        output_h5: Union[str, Path] = None,
        dataset_name: str = 'cxr'
    ) -> str:
        """
        Convert a single image to H5 format

        Args:
            image_path: Path to the image file
            output_h5: Path for output H5 file (if None, creates temp file)
            dataset_name: Name of the dataset in H5 file

        Returns:
            Path to the created H5 file
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)

        # Add batch dimension: (1, height, width)
        img_array = img_array[np.newaxis, ...]

        # Create output path if not provided
        if output_h5 is None:
            # Create temp file
            fd, output_h5 = tempfile.mkstemp(suffix='.h5')
            os.close(fd)

        # Save to H5 file
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset(dataset_name, data=img_array, compression='gzip')

        return str(output_h5)

    def convert_multiple_images(
        self,
        image_paths: List[Union[str, Path]],
        output_h5: Union[str, Path],
        dataset_name: str = 'cxr'
    ) -> str:
        """
        Convert multiple images to a single H5 file

        Args:
            image_paths: List of paths to image files
            output_h5: Path for output H5 file
            dataset_name: Name of the dataset in H5 file

        Returns:
            Path to the created H5 file
        """
        # Preprocess all images
        images = []
        for img_path in image_paths:
            img_array = self.preprocess_image(img_path)
            images.append(img_array)

        # Stack into single array: (batch, height, width)
        images_array = np.stack(images, axis=0)

        # Save to H5 file
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset(dataset_name, data=images_array, compression='gzip')

        return str(output_h5)

    def load_h5_images(
        self,
        h5_path: Union[str, Path],
        dataset_name: str = 'cxr'
    ) -> np.ndarray:
        """
        Load images from H5 file

        Args:
            h5_path: Path to H5 file
            dataset_name: Name of the dataset in H5 file

        Returns:
            Images as numpy array
        """
        with h5py.File(h5_path, 'r') as f:
            images = f[dataset_name][:]

        return images

    def verify_h5_file(self, h5_path: Union[str, Path]) -> dict:
        """
        Verify and inspect an H5 file

        Args:
            h5_path: Path to H5 file

        Returns:
            Dictionary with file information
        """
        info = {}

        with h5py.File(h5_path, 'r') as f:
            # List all datasets
            info['datasets'] = list(f.keys())

            # Get shape and dtype for each dataset
            info['details'] = {}
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                info['details'][dataset_name] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'size_mb': dataset.nbytes / (1024 * 1024)
                }

        return info


# Convenience functions
def image_to_h5(
    image_path: Union[str, Path],
    output_h5: Union[str, Path] = None,
    image_size: tuple = (224, 224)
) -> str:
    """
    Quick conversion of a single image to H5

    Args:
        image_path: Path to the image file
        output_h5: Path for output H5 file (optional)
        image_size: Target image size

    Returns:
        Path to the created H5 file
    """
    converter = H5Converter(image_size=image_size)
    return converter.convert_single_image(image_path, output_h5)


def images_to_h5(
    image_paths: List[Union[str, Path]],
    output_h5: Union[str, Path],
    image_size: tuple = (224, 224)
) -> str:
    """
    Quick conversion of multiple images to H5

    Args:
        image_paths: List of paths to image files
        output_h5: Path for output H5 file
        image_size: Target image size

    Returns:
        Path to the created H5 file
    """
    converter = H5Converter(image_size=image_size)
    return converter.convert_multiple_images(image_paths, output_h5)


if __name__ == "__main__":
    # Test the converter
    import sys

    if len(sys.argv) < 2:
        print("Usage: python h5_converter.py <image_path> [output.h5]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_h5 = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Converting {image_path} to H5 format...")

    converter = H5Converter()
    output_path = converter.convert_single_image(image_path, output_h5)

    print(f"✅ Created: {output_path}")

    # Verify
    info = converter.verify_h5_file(output_path)
    print(f"\nH5 File Info:")
    print(f"   Datasets: {info['datasets']}")
    for name, details in info['details'].items():
        print(f"   {name}:")
        print(f"      Shape: {details['shape']}")
        print(f"      Type: {details['dtype']}")
        print(f"      Size: {details['size_mb']:.2f} MB")
