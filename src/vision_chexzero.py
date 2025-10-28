"""
CheXzero Vision Analyzer
Medical-grade X-ray analysis using Stanford's CheXzero model

Author: MedAssist Copilot Team
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time
from typing import Dict, List, Any, Union
import tempfile
import os

# Add CheXzero to path
CHEXZERO_PATH = Path(__file__).parent.parent / "models" / "CheXzero"
sys.path.insert(0, str(CHEXZERO_PATH))

from src.h5_converter import H5Converter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheXzeroAnalyzer:
    """
    CheXzero-based vision analyzer for medical-grade X-ray analysis

    Uses Stanford's CheXzero model to detect pathologies with expert-level accuracy.
    """

    # Standard pathologies detected by CheXzero
    PATHOLOGIES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Support Devices",
        "No Finding"
    ]

    # Contrasting templates for zero-shot inference
    CXR_PAIR_TEMPLATE = (
        "Findings consistent with {}",
        "No evidence of {}"
    )

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        threshold: float = 0.5,
        image_size: tuple = (224, 224)
    ):
        """
        Initialize CheXzero analyzer

        Args:
            model_dir: Directory containing model weights (.pt files)
            threshold: Probability threshold for pathology detection (0-1)
            image_size: Image size for preprocessing
        """
        if model_dir is None:
            model_dir = CHEXZERO_PATH / "checkpoints" / "chexzero_weights"

        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.image_size = image_size

        # Initialize H5 converter
        self.h5_converter = H5Converter(image_size=image_size)

        # Load model checkpoints
        self.model_paths = self._find_model_weights()

        if not self.model_paths:
            raise FileNotFoundError(
                f"No CheXzero model weights found in {model_dir}\n"
                f"Please download from: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno\n"
                f"Or run: python download_chexzero_weights.py"
            )

        logger.info(f"Initialized CheXzero with {len(self.model_paths)} model checkpoint(s)")

        # Lazy loading of CheXzero modules (heavy imports)
        self._zero_shot = None

    def _find_model_weights(self) -> List[Path]:
        """Find all model weight files"""
        return list(self.model_dir.glob("*.pt"))

    @property
    def zero_shot(self):
        """Lazy load CheXzero's zero_shot module"""
        if self._zero_shot is None:
            try:
                import zero_shot
                self._zero_shot = zero_shot
                logger.info("Loaded CheXzero zero_shot module")
            except ImportError as e:
                raise ImportError(
                    f"Failed to import CheXzero's zero_shot module: {e}\n"
                    f"Make sure CheXzero is properly installed at: {CHEXZERO_PATH}"
                )
        return self._zero_shot

    def analyze_xray(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze X-ray using CheXzero

        Args:
            image_path: Path to the X-ray image
            return_probabilities: Whether to return full probability array

        Returns:
            Dictionary with analysis results:
            - is_normal: bool
            - detected_pathologies: List[Dict]
            - confidence: str (HIGH/MODERATE/LOW)
            - processing_time: float
            - all_probabilities: List[float] (optional)
        """
        start_time = time.time()

        logger.info(f"Analyzing X-ray with CheXzero: {image_path}")

        try:
            # Convert image to H5 format
            logger.info("Converting image to H5 format...")
            h5_path = self._prepare_image(image_path)

            # Run ensemble inference
            logger.info("Running CheXzero inference...")
            predictions = self._run_inference(h5_path)

            # Parse results
            logger.info("Parsing results...")
            results = self._parse_predictions(predictions)

            # Add metadata
            results['processing_time'] = time.time() - start_time

            if return_probabilities:
                results['all_probabilities'] = predictions.tolist()

            logger.info(f"Analysis complete in {results['processing_time']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Error analyzing X-ray: {e}")
            raise

        finally:
            # Clean up temp H5 file
            if 'h5_path' in locals() and h5_path and Path(h5_path).exists():
                try:
                    os.unlink(h5_path)
                except:
                    pass

    def _prepare_image(self, image_path: Union[str, Path]) -> str:
        """
        Convert image to H5 format

        Args:
            image_path: Path to the image

        Returns:
            Path to the H5 file
        """
        return self.h5_converter.convert_single_image(
            image_path,
            output_h5=None  # Use temp file
        )

    def _run_inference(self, h5_path: str) -> np.ndarray:
        """
        Run CheXzero ensemble inference

        Args:
            h5_path: Path to H5 file containing the image

        Returns:
            Numpy array of probabilities for each pathology
        """
        # Import ensemble_models from CheXzero
        from zero_shot import ensemble_models

        # Run ensemble inference
        predictions, y_pred_avg = ensemble_models(
            model_paths=[str(p) for p in self.model_paths],
            cxr_filepath=h5_path,
            cxr_labels=self.PATHOLOGIES,
            cxr_pair_template=self.CXR_PAIR_TEMPLATE,
            cache_dir=None  # Disable caching for single predictions
        )

        # Return average predictions across ensemble
        return y_pred_avg[0]  # Shape: (num_pathologies,)

    def _parse_predictions(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Parse model predictions into structured format

        Args:
            probabilities: Array of probabilities for each pathology

        Returns:
            Dictionary with parsed results
        """
        detected_pathologies = []

        for i, pathology in enumerate(self.PATHOLOGIES):
            prob = float(probabilities[i])

            # Only include pathologies above threshold
            if prob > self.threshold:
                # Determine confidence level
                if prob >= 0.8:
                    confidence = "HIGH"
                elif prob >= 0.6:
                    confidence = "MODERATE"
                else:
                    confidence = "LOW"

                detected_pathologies.append({
                    "pathology": pathology,
                    "confidence": confidence,
                    "probability": prob
                })

        # Sort by probability (highest first)
        detected_pathologies.sort(key=lambda x: x['probability'], reverse=True)

        # Determine if image is normal
        is_normal = self._is_normal(detected_pathologies, probabilities)

        # Determine overall confidence
        overall_confidence = self._get_overall_confidence(detected_pathologies)

        return {
            "is_normal": is_normal,
            "detected_pathologies": detected_pathologies,
            "confidence": overall_confidence,
            "processing_time": 0  # Will be set by analyze_xray
        }

    def _is_normal(
        self,
        detected_pathologies: List[Dict],
        probabilities: np.ndarray
    ) -> bool:
        """
        Determine if X-ray is normal

        Args:
            detected_pathologies: List of detected pathologies
            probabilities: Full probability array

        Returns:
            True if normal, False otherwise
        """
        # Check "No Finding" probability
        no_finding_idx = self.PATHOLOGIES.index("No Finding")
        no_finding_prob = float(probabilities[no_finding_idx])

        # If "No Finding" has high probability and few other pathologies detected
        if no_finding_prob > 0.7 and len(detected_pathologies) <= 1:
            return True

        # If no significant pathologies detected
        if len(detected_pathologies) == 0:
            return True

        # If only "No Finding" detected
        if len(detected_pathologies) == 1 and detected_pathologies[0]['pathology'] == "No Finding":
            return True

        return False

    def _get_overall_confidence(self, detected_pathologies: List[Dict]) -> str:
        """
        Get overall confidence level

        Args:
            detected_pathologies: List of detected pathologies

        Returns:
            Confidence level: HIGH, MODERATE, or LOW
        """
        if not detected_pathologies:
            return "HIGH"  # High confidence it's normal

        # Get highest probability
        max_prob = max(p['probability'] for p in detected_pathologies)

        if max_prob >= 0.8:
            return "HIGH"
        elif max_prob >= 0.6:
            return "MODERATE"
        else:
            return "LOW"

    def generate_findings_summary(self, analysis_result: Dict[str, Any]) -> str:
        """
        Generate a textual summary of findings

        Args:
            analysis_result: Result from analyze_xray()

        Returns:
            Text summary of findings
        """
        pathologies = analysis_result['detected_pathologies']
        is_normal = analysis_result['is_normal']
        confidence = analysis_result['confidence']

        if is_normal:
            return (
                f"NORMAL chest X-ray\n"
                f"No significant pathologies detected\n"
                f"Analysis confidence: {confidence}"
            )
        else:
            lines = [
                f"ABNORMAL chest X-ray",
                f"Detected pathologies: {len(pathologies)}"
            ]

            for pathology in pathologies[:5]:  # Top 5
                lines.append(
                    f"  - {pathology['pathology']}: "
                    f"{pathology['probability']:.3f} ({pathology['confidence']} confidence)"
                )

            lines.append(f"Analysis confidence: {confidence}")

            return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision_chexzero.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Initializing CheXzero analyzer...")
    analyzer = CheXzeroAnalyzer()

    print(f"\nAnalyzing: {image_path}")
    result = analyzer.analyze_xray(image_path)

    print("\n" + "=" * 70)
    print("CheXzero Analysis Results")
    print("=" * 70)
    print(analyzer.generate_findings_summary(result))
    print("=" * 70)
