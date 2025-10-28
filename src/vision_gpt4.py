"""
GPT-4 Vision Analyzer for Medical Imaging
Uses GPT-4o with vision capabilities for medical X-ray analysis

This module provides a medical-grade vision analysis system that can:
- Detect pathologies in chest X-rays
- Generate structured medical findings
- Provide confidence assessments
- Work with standard image formats (PNG, JPG)

Author: MedAssist Copilot Team
Created: Week 4.5 - Medical Vision Upgrade
"""

import os
import base64
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import json

import openai
from openai import OpenAI

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT4VisionAnalyzer:
    """
    Medical imaging analyzer using GPT-4 Vision (GPT-4o)

    This analyzer uses OpenAI's GPT-4o model with vision capabilities to:
    1. Analyze chest X-ray images
    2. Detect pathologies and abnormalities
    3. Generate structured medical findings
    4. Provide confidence assessments for findings

    Attributes:
        model_name (str): GPT-4 vision model to use
        client (OpenAI): OpenAI API client
        max_tokens (int): Maximum tokens for response
        temperature (float): Sampling temperature
    """

    # Standard pathologies to check for (CheXpert-based)
    PATHOLOGIES = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Pleural Effusion',
        'Pneumonia',
        'Pneumothorax',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
        'Fracture',
        'Lung Lesion',
        'Support Devices',
    ]

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,  # Lower temperature for consistent medical analysis
    ):
        """
        Initialize GPT-4 Vision Analyzer

        Args:
            model_name: GPT-4 vision model (default: gpt-4o)
            api_key: OpenAI API key (uses config if not provided)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0.0-1.0)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = api_key or config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env")

        self.client = OpenAI(api_key=api_key)

        # Statistics tracking
        self.stats = {
            'images_analyzed': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'api_calls': 0,
        }

        logger.info(f"Initialized GPT-4 Vision Analyzer with model: {model_name}")

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 for API transmission

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_xray(
        self,
        image_path: Union[str, Path],
        include_pathology_detection: bool = True,
        include_structured_report: bool = True,
    ) -> Dict:
        """
        Analyze chest X-ray image using GPT-4 Vision

        Args:
            image_path: Path to chest X-ray image
            include_pathology_detection: Whether to detect specific pathologies
            include_structured_report: Whether to generate structured findings

        Returns:
            Dictionary containing:
                - raw_analysis: Full text analysis
                - detected_pathologies: List of detected pathologies with confidence
                - findings: Structured medical findings
                - is_normal: Whether the X-ray appears normal
                - confidence: Overall confidence in analysis
                - processing_time: Time taken for analysis
        """
        start_time = time.time()

        try:
            # Encode image
            base64_image = self.encode_image(image_path)

            # Create comprehensive medical analysis prompt
            prompt = self._create_medical_analysis_prompt(
                include_pathology_detection=include_pathology_detection,
                include_structured_report=include_structured_report
            )

            # Call GPT-4 Vision API
            logger.info(f"Analyzing X-ray: {image_path}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert radiologist with years of experience in interpreting chest X-rays. Provide accurate, professional medical analysis."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # High detail for medical images
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Extract response
            analysis_text = response.choices[0].message.content

            # Parse structured response
            result = self._parse_medical_response(analysis_text)

            # Add metadata
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['model_used'] = self.model_name
            result['image_path'] = str(image_path)

            # Update statistics
            self.stats['images_analyzed'] += 1
            self.stats['total_time'] += processing_time
            self.stats['average_time'] = self.stats['total_time'] / self.stats['images_analyzed']
            self.stats['api_calls'] += 1

            logger.info(f"Analysis completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error analyzing X-ray: {e}")
            raise

    def _create_medical_analysis_prompt(
        self,
        include_pathology_detection: bool = True,
        include_structured_report: bool = True
    ) -> str:
        """Create comprehensive medical analysis prompt"""

        prompt = """Analyze this chest X-ray image as an expert radiologist would.

Please provide a comprehensive analysis following this structure:

"""

        if include_pathology_detection:
            pathology_list = ', '.join(self.PATHOLOGIES)
            prompt += f"""
1. PATHOLOGY DETECTION:
   For each of these pathologies, indicate if present and your confidence level:
   {pathology_list}

   Format:
   - Pathology Name: [PRESENT/ABSENT] (Confidence: HIGH/MODERATE/LOW)
   - Brief justification for each positive finding

"""

        if include_structured_report:
            prompt += """
2. DETAILED FINDINGS:
   Systematically evaluate:
   - Cardiac silhouette (size, contour, position)
   - Lung fields (both right and left, clarity, any opacities)
   - Pleural spaces (effusions, pneumothorax)
   - Mediastinum (width, contour, position)
   - Bones and soft tissues (fractures, lesions)
   - Support devices if present (tubes, lines, pacemakers)

3. CLINICAL IMPRESSION:
   - Primary diagnosis or most likely diagnosis
   - Differential diagnoses if applicable
   - Overall severity assessment

4. NORMAL vs ABNORMAL:
   - State clearly: NORMAL or ABNORMAL
   - If abnormal, list priority findings

5. CONFIDENCE ASSESSMENT:
   - Overall confidence in this analysis (HIGH/MODERATE/LOW)
   - Any limitations or areas of uncertainty

Please be specific, use proper medical terminology, and provide clinically relevant observations.
"""

        return prompt

    def _parse_medical_response(self, analysis_text: str) -> Dict:
        """
        Parse the medical analysis response into structured format

        Args:
            analysis_text: Raw text response from GPT-4 Vision

        Returns:
            Structured dictionary with parsed findings
        """
        result = {
            'raw_analysis': analysis_text,
            'detected_pathologies': [],
            'findings': '',
            'impression': '',
            'is_normal': False,
            'confidence': 'MODERATE',
            'severity': 'UNKNOWN'
        }

        # Parse for normal/abnormal
        analysis_lower = analysis_text.lower()
        if 'normal' in analysis_lower and 'abnormal' not in analysis_lower.split('normal')[0]:
            result['is_normal'] = True

        # Extract confidence level
        if 'high' in analysis_lower and 'confidence' in analysis_lower:
            result['confidence'] = 'HIGH'
        elif 'low' in analysis_lower and 'confidence' in analysis_lower:
            result['confidence'] = 'LOW'

        # Parse detected pathologies
        for pathology in self.PATHOLOGIES:
            if pathology.lower() in analysis_lower:
                # Check if it's marked as present
                # Look for context around the pathology name
                for line in analysis_text.split('\n'):
                    if pathology.lower() in line.lower():
                        if 'present' in line.lower() and 'absent' not in line.lower():
                            confidence = 'MODERATE'
                            if 'high' in line.lower():
                                confidence = 'HIGH'
                            elif 'low' in line.lower():
                                confidence = 'LOW'

                            result['detected_pathologies'].append({
                                'pathology': pathology,
                                'status': 'PRESENT',
                                'confidence': confidence,
                                'context': line.strip()
                            })
                            break

        # Extract sections
        sections = analysis_text.split('\n\n')
        for i, section in enumerate(sections):
            section_lower = section.lower()
            if 'findings' in section_lower or 'detailed findings' in section_lower:
                result['findings'] = section
            elif 'impression' in section_lower or 'clinical impression' in section_lower:
                result['impression'] = section

        # If no specific sections found, use first part as findings
        if not result['findings']:
            result['findings'] = sections[0] if sections else analysis_text[:500]

        return result

    def generate_findings_summary(self, analysis_result: Dict) -> str:
        """
        Generate a concise findings summary for downstream LLM processing

        Args:
            analysis_result: Result from analyze_xray()

        Returns:
            Concise summary string suitable for LLM input
        """
        summary_parts = []

        # Add normal/abnormal status
        status = "NORMAL chest X-ray" if analysis_result['is_normal'] else "ABNORMAL chest X-ray"
        summary_parts.append(status)

        # Add detected pathologies
        if analysis_result['detected_pathologies']:
            pathologies = [
                f"{p['pathology']} ({p['confidence']} confidence)"
                for p in analysis_result['detected_pathologies']
            ]
            summary_parts.append("Detected pathologies: " + ", ".join(pathologies))

        # Add key findings
        if analysis_result['findings']:
            # Extract first few sentences
            findings_preview = analysis_result['findings'][:300]
            summary_parts.append(f"Key findings: {findings_preview}")

        # Add confidence
        summary_parts.append(f"Analysis confidence: {analysis_result['confidence']}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'images_analyzed': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'api_calls': 0,
        }


# Convenience function for quick analysis
def analyze_chest_xray(image_path: Union[str, Path], model: str = "gpt-4o") -> Dict:
    """
    Quick convenience function to analyze a chest X-ray

    Args:
        image_path: Path to X-ray image
        model: GPT-4 vision model to use

    Returns:
        Analysis result dictionary
    """
    analyzer = GPT4VisionAnalyzer(model_name=model)
    return analyzer.analyze_xray(image_path)


# Testing code
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("MedAssist Copilot - GPT-4 Vision Analyzer Test")
    print("=" * 70)

    # Initialize analyzer
    print("\n1️⃣  Initializing GPT-4 Vision Analyzer...")
    try:
        analyzer = GPT4VisionAnalyzer(model_name="gpt-4o")
        print(f"   ✅ Initialized successfully")
        print(f"   Model: {analyzer.model_name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)

    # Find test images
    print("\n2️⃣  Finding test images...")
    data_dir = Path("data/raw")

    # Try to find sample images
    normal_dir = data_dir / "NORMAL"
    pneumonia_dir = data_dir / "PNEUMONIA"

    test_images = []
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg"))[:2]
        test_images.extend([(img, "NORMAL") for img in normal_images])

    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))[:2]
        test_images.extend([(img, "PNEUMONIA") for img in pneumonia_images])

    if not test_images:
        print("   ❌ No test images found in data/raw/")
        sys.exit(1)

    print(f"   Found {len(test_images)} test images")

    # Analyze images
    print("\n3️⃣  Analyzing chest X-rays...")
    for i, (image_path, true_label) in enumerate(test_images, 1):
        print(f"\n   Image {i}: {image_path.name} (Actual: {true_label})")

        try:
            result = analyzer.analyze_xray(image_path)

            print(f"   Status: {'NORMAL' if result['is_normal'] else 'ABNORMAL'}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")

            if result['detected_pathologies']:
                print(f"   Detected pathologies:")
                for pathology in result['detected_pathologies']:
                    print(f"      - {pathology['pathology']}: {pathology['confidence']}")
            else:
                print(f"   No significant pathologies detected")

            # Show summary
            summary = analyzer.generate_findings_summary(result)
            print(f"\n   Summary for LLM:")
            print(f"   {summary[:200]}...")

        except Exception as e:
            print(f"   ❌ Error analyzing image: {e}")

    # Show statistics
    print("\n4️⃣  Analysis Statistics:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("GPT-4 Vision analysis complete!")
    print("=" * 70)
