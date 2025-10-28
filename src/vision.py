"""
Vision Module for MedAssist Copilot
Handles vision-language model integration for X-ray image analysis
Supports multiple backends:
  - BLIP-2: General-purpose vision-language model
  - GPT-4 Vision: Medical-grade multimodal analysis (recommended)
  - Hybrid: Uses GPT-4 Vision when available, falls back to BLIP-2
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GPT-4 Vision analyzer
try:
    from src.vision_gpt4 import GPT4VisionAnalyzer
    GPT4_VISION_AVAILABLE = True
    logger.info("GPT-4 Vision module available")
except ImportError as e:
    GPT4_VISION_AVAILABLE = False
    logger.warning(f"GPT-4 Vision module not available: {e}")


class VisionAnalyzer:
    """
    Vision model wrapper for chest X-ray analysis
    Supports BLIP-2 and other vision-language models
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = "models"
    ):
        """
        Initialize the vision analyzer

        Args:
            model_name: Name of the vision model (defaults to config.VISION_MODEL_NAME)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name or config.VISION_MODEL_NAME
        self.cache_dir = cache_dir

        # Determine device
        if device is None:
            self.device = config.VISION_MODEL_DEVICE
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        logger.info(f"Initializing Vision Analyzer with {self.model_name} on {self.device}")

        # Model and processor (loaded lazily)
        self.model = None
        self.processor = None
        self.model_loaded = False

        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'cache_hits': 0
        }

        # Caption cache (for repeated images)
        self.caption_cache = {}

    def load_model(self) -> bool:
        """
        Load the vision model and processor

        Returns:
            True if successful, False otherwise
        """
        if self.model_loaded:
            logger.info("Model already loaded")
            return True

        try:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()

            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)

            # Load processor and model based on model type
            if "blip" in self.model_name.lower():
                self.processor = BlipProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            else:
                # Generic AutoModel approach for other models
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            load_time = time.time() - start_time
            self.model_loaded = True

            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Parameters: {self._count_parameters():,}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False

    def _count_parameters(self) -> int:
        """Count the number of parameters in the model"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def generate_caption(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 100,
        num_beams: int = 5,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Generate a caption for an X-ray image

        Args:
            image: Path to image file or PIL Image object
            prompt: Optional text prompt to guide generation
            max_length: Maximum length of generated caption
            num_beams: Number of beams for beam search
            use_cache: Whether to use cached captions

        Returns:
            Generated caption string or None if failed
        """
        # Load model if not already loaded
        if not self.model_loaded:
            if not self.load_model():
                return None

        try:
            # Load image if path provided
            if isinstance(image, str):
                image_key = image
                if use_cache and image_key in self.caption_cache:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Using cached caption for {Path(image).name}")
                    return self.caption_cache[image_key]

                image = Image.open(image).convert('RGB')
            else:
                image_key = None

            start_time = time.time()

            # Process image
            if prompt:
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )

            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Update statistics
            process_time = time.time() - start_time
            self.stats['images_processed'] += 1
            self.stats['total_time'] += process_time
            self.stats['average_time'] = self.stats['total_time'] / self.stats['images_processed']

            # Cache result
            if image_key and use_cache:
                self.caption_cache[image_key] = caption

            logger.info(f"Generated caption in {process_time:.2f}s")
            logger.debug(f"Caption: {caption}")

            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return None

    def generate_medical_description(
        self,
        image: Union[str, Image.Image],
        condition: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a medical description of an X-ray image

        Args:
            image: Path to image file or PIL Image object
            condition: Optional condition to focus on

        Returns:
            Medical description or None if failed
        """
        # Create medical-focused prompt
        if condition:
            prompt = f"A chest X-ray showing {condition}:"
        else:
            prompt = "A detailed medical description of this chest X-ray:"

        caption = self.generate_caption(
            image=image,
            prompt=prompt,
            max_length=150,
            num_beams=5
        )

        return caption

    def batch_generate_captions(
        self,
        images: List[Union[str, Image.Image]],
        prompts: Optional[List[str]] = None,
        max_length: int = 100,
        num_beams: int = 5
    ) -> List[Optional[str]]:
        """
        Generate captions for multiple images

        Args:
            images: List of image paths or PIL Images
            prompts: Optional list of prompts (one per image)
            max_length: Maximum length of generated captions
            num_beams: Number of beams for beam search

        Returns:
            List of generated captions
        """
        if not self.model_loaded:
            if not self.load_model():
                return [None] * len(images)

        captions = []

        for i, image in enumerate(images):
            prompt = prompts[i] if prompts and i < len(prompts) else None

            caption = self.generate_caption(
                image=image,
                prompt=prompt,
                max_length=max_length,
                num_beams=num_beams
            )

            captions.append(caption)

            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images")

        return captions

    def analyze_xray(
        self,
        image: Union[str, Image.Image],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive X-ray analysis with multiple descriptions

        Args:
            image: Path to image file or PIL Image object
            detailed: Whether to generate detailed descriptions

        Returns:
            Dictionary with analysis results
        """
        result = {
            'success': False,
            'image_path': str(image) if isinstance(image, str) else 'PIL_Image',
            'captions': {},
            'timestamp': time.time()
        }

        try:
            # Generate base caption
            logger.info("Generating base caption...")
            base_caption = self.generate_caption(image)
            result['captions']['base'] = base_caption

            if detailed:
                # Generate medical description
                logger.info("Generating medical description...")
                medical_desc = self.generate_medical_description(image)
                result['captions']['medical'] = medical_desc

                # Generate focused descriptions for specific conditions
                conditions = ['consolidation', 'effusion', 'cardiomegaly']
                for condition in conditions:
                    logger.info(f"Checking for {condition}...")
                    desc = self.generate_caption(
                        image,
                        prompt=f"Is there evidence of {condition} in this chest X-ray?",
                        max_length=50
                    )
                    result['captions'][f'check_{condition}'] = desc

            result['success'] = True
            logger.info("✅ X-ray analysis complete")

        except Exception as e:
            logger.error(f"Error during X-ray analysis: {str(e)}")
            result['error'] = str(e)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()

    def clear_cache(self):
        """Clear the caption cache"""
        self.caption_cache.clear()
        logger.info("Caption cache cleared")

    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.model_loaded = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")


# Utility functions

def create_vision_analyzer(backend: Optional[str] = None) -> Union[VisionAnalyzer, 'GPT4VisionAnalyzer']:
    """
    Factory function to create appropriate vision analyzer based on backend

    Args:
        backend: Vision backend to use:
            - 'blip': Use BLIP-2 (default, fast, CPU-friendly)
            - 'gpt4': Use GPT-4 Vision (recommended, medical-grade)
            - 'auto': Auto-select (GPT-4 if available, else BLIP-2)
            - None: Use config.VISION_BACKEND

    Returns:
        Initialized vision analyzer instance
    """
    # Determine backend
    if backend is None:
        backend = getattr(config, 'VISION_BACKEND', 'blip')

    backend = backend.lower()

    # Auto-selection
    if backend == 'auto':
        if GPT4_VISION_AVAILABLE and config.OPENAI_API_KEY:
            backend = 'gpt4'
            logger.info("Auto-selected GPT-4 Vision backend")
        else:
            backend = 'blip'
            logger.info("Auto-selected BLIP-2 backend (GPT-4 Vision not available)")

    # Create analyzer
    if backend == 'gpt4':
        if not GPT4_VISION_AVAILABLE:
            logger.warning("GPT-4 Vision not available, falling back to BLIP-2")
            return VisionAnalyzer()

        if not config.OPENAI_API_KEY:
            logger.warning("OpenAI API key not found, falling back to BLIP-2")
            return VisionAnalyzer()

        logger.info("Creating GPT-4 Vision analyzer")
        return GPT4VisionAnalyzer(model_name=getattr(config, 'GPT4_VISION_MODEL', 'gpt-4o'))

    elif backend == 'blip':
        logger.info("Creating BLIP-2 analyzer")
        return VisionAnalyzer()

    else:
        logger.warning(f"Unknown backend '{backend}', defaulting to BLIP-2")
        return VisionAnalyzer()


def quick_caption(image_path: str, model_name: Optional[str] = None) -> Optional[str]:
    """
    Quick function to generate a caption for an image

    Args:
        image_path: Path to the image
        model_name: Optional model name

    Returns:
        Generated caption or None
    """
    analyzer = VisionAnalyzer(model_name=model_name)
    return analyzer.generate_caption(image_path)


# Main execution for testing
if __name__ == "__main__":
    print("=" * 70)
    print("MedAssist Copilot - Vision Module Test")
    print("=" * 70)

    # Initialize analyzer
    print("\n1️⃣  Initializing Vision Analyzer...")
    analyzer = VisionAnalyzer()

    # Find test images
    data_dir = Path("data/raw")
    test_images = []

    # Look for NORMAL images
    normal_dir = data_dir / "NORMAL"
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg"))[:2]
        test_images.extend(normal_images)

    # Look for PNEUMONIA images
    pneumonia_dir = data_dir / "PNEUMONIA"
    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))[:2]
        test_images.extend(pneumonia_images)

    if not test_images:
        print("   ⚠️  No test images found in data/raw/NORMAL or data/raw/PNEUMONIA")
        print("   Please add some X-ray images first")
        exit(1)

    print(f"   Found {len(test_images)} test images")

    # Load model
    print("\n2️⃣  Loading vision model...")
    if not analyzer.load_model():
        print("   ❌ Failed to load model")
        exit(1)

    # Test caption generation
    print("\n3️⃣  Testing caption generation...")
    for i, img_path in enumerate(test_images, 1):
        print(f"\n   Image {i}: {img_path.name}")
        caption = analyzer.generate_caption(str(img_path))

        if caption:
            print(f"   Caption: {caption}")
        else:
            print("   ❌ Failed to generate caption")

    # Show statistics
    print("\n4️⃣  Processing Statistics:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("Vision module test complete!")
    print("=" * 70)
