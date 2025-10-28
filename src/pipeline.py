"""
End-to-End Pipeline for MedAssist Copilot
Connects Vision Model → RAG → LLM → Report Generation
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import time
from PIL import Image

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import VisionAnalyzer, create_vision_analyzer
from src.llm_processor import LLMProcessor
from src.report_manager import ReportManager
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerationPipeline:
    """
    Complete pipeline for generating radiology reports from X-ray images
    Integrates vision analysis, RAG retrieval, and LLM generation
    """

    def __init__(
        self,
        vision_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        use_rag: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the pipeline

        Args:
            vision_backend: Vision backend ('blip', 'gpt4', 'auto', or None for config default)
            llm_model: LLM model name (defaults to config)
            use_rag: Whether to use RAG for context
            api_key: OpenAI API key (defaults to config)
        """
        logger.info("Initializing Report Generation Pipeline...")

        # Initialize components
        self.use_rag = use_rag and config.ENABLE_RAG

        # Vision Analyzer (using factory function)
        logger.info("Loading vision analyzer...")
        self.vision_analyzer = create_vision_analyzer(backend=vision_backend)
        self.vision_backend = vision_backend or config.VISION_BACKEND
        logger.info(f"   Using vision backend: {self.vision_backend}")

        # LLM Processor
        logger.info("Initializing LLM processor...")
        self.llm_processor = LLMProcessor(
            api_key=api_key,
            model_name=llm_model
        )

        # Report Manager (for RAG)
        if self.use_rag:
            logger.info("Loading report database for RAG...")
            self.report_manager = ReportManager()
        else:
            self.report_manager = None

        # Pipeline statistics
        self.stats = {
            'total_reports': 0,
            'successful_reports': 0,
            'failed_reports': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'vision_time': 0.0,
            'rag_time': 0.0,
            'llm_time': 0.0
        }

        logger.info("✅ Pipeline initialized successfully!")

    def generate_report(
        self,
        image: Union[str, Image.Image],
        patient_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        indication: Optional[str] = None,
        detailed_vision: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a complete radiology report from an X-ray image

        Args:
            image: Path to X-ray image or PIL Image object
            patient_id: Patient identifier
            age: Patient age
            gender: Patient gender
            indication: Clinical indication for the study
            detailed_vision: Whether to generate detailed vision analysis

        Returns:
            Dictionary with report and metadata
        """
        start_time = time.time()
        self.stats['total_reports'] += 1

        result = {
            'success': False,
            'patient_id': patient_id,
            'image_path': str(image) if isinstance(image, str) else 'PIL_Image',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            logger.info("=" * 70)
            logger.info("Starting report generation pipeline...")
            logger.info("=" * 70)

            # Step 1: Vision Analysis
            logger.info("\n📸 Step 1/3: Analyzing X-ray image...")
            vision_start = time.time()

            # Handle different vision backends
            if self.vision_backend == 'gpt4':
                # GPT-4 Vision provides structured medical analysis
                vision_result = self.vision_analyzer.analyze_xray(image)
                caption = self.vision_analyzer.generate_findings_summary(vision_result)

                # Store detailed findings for potential UI display
                result['vision_details'] = {
                    'raw_analysis': vision_result.get('raw_analysis', ''),
                    'detected_pathologies': vision_result.get('detected_pathologies', []),
                    'is_normal': vision_result.get('is_normal', False),
                    'confidence': vision_result.get('confidence', 'UNKNOWN')
                }
            else:
                # BLIP-2 provides simple captions
                if detailed_vision:
                    vision_result = self.vision_analyzer.analyze_xray(image, detailed=True)
                    caption = vision_result['captions'].get('medical',
                             vision_result['captions'].get('base', ''))
                else:
                    caption = self.vision_analyzer.generate_medical_description(image)

                result['vision_details'] = {'caption_only': True}

            vision_time = time.time() - vision_start
            self.stats['vision_time'] += vision_time

            if not caption:
                logger.error("❌ Failed to generate image caption")
                result['error'] = "Vision analysis failed"
                self.stats['failed_reports'] += 1
                return result

            logger.info(f"   ✅ Image analyzed in {vision_time:.2f}s")
            logger.info(f"   Caption preview: {caption[:150]}...")
            result['vision_caption'] = caption
            result['vision_time'] = vision_time
            result['vision_backend'] = self.vision_backend

            # Step 2: RAG - Retrieve Prior Reports
            rag_context = None
            if self.use_rag and patient_id and self.report_manager:
                logger.info("\n🔍 Step 2/3: Retrieving prior reports (RAG)...")
                rag_start = time.time()

                prior_reports = self.report_manager.get_report_by_patient_id(patient_id)

                if prior_reports:
                    # Use the most recent report
                    latest_report = prior_reports[0]
                    rag_context = f"""Previous Report ({latest_report['date']}):
Findings: {latest_report['report']['findings'][:200]}...
Impression: {latest_report['report']['impression'][:100]}..."""

                    logger.info(f"   ✅ Retrieved {len(prior_reports)} prior report(s)")
                else:
                    logger.info("   ℹ️  No prior reports found for this patient")
                    rag_context = "No prior reports available for this patient."

                rag_time = time.time() - rag_start
                self.stats['rag_time'] += rag_time
                result['rag_time'] = rag_time
                result['prior_reports_count'] = len(prior_reports)
            else:
                logger.info("\n⏭️  Step 2/3: Skipping RAG (disabled or no patient ID)")
                rag_context = "No prior reports available."
                result['rag_time'] = 0.0
                result['prior_reports_count'] = 0

            # Step 3: LLM Report Generation
            logger.info("\n🤖 Step 3/3: Generating radiology report...")
            llm_start = time.time()

            llm_result = self.llm_processor.generate_report(
                vision_caption=caption,
                patient_id=patient_id,
                age=age,
                gender=gender,
                rag_context=rag_context
            )

            llm_time = time.time() - llm_start
            self.stats['llm_time'] += llm_time

            if not llm_result or not llm_result['success']:
                logger.error("❌ Failed to generate report with LLM")
                result['error'] = "LLM generation failed"
                self.stats['failed_reports'] += 1
                return result

            logger.info(f"   ✅ Report generated in {llm_time:.2f}s")
            logger.info(f"   Tokens used: {llm_result['metadata']['tokens_used']}")

            # Combine results
            result.update({
                'success': True,
                'report_text': llm_result['report_text'],
                'report_sections': llm_result['sections'],
                'llm_time': llm_time,
                'tokens_used': llm_result['metadata']['tokens_used']
            })

            # Calculate total time
            total_time = time.time() - start_time
            result['total_time'] = total_time

            self.stats['successful_reports'] += 1
            self.stats['total_time'] += total_time
            self.stats['average_time'] = self.stats['total_time'] / self.stats['successful_reports']

            logger.info("\n" + "=" * 70)
            logger.info(f"✅ Pipeline completed successfully in {total_time:.2f}s")
            logger.info("=" * 70)

            return result

        except Exception as e:
            logger.error(f"\n❌ Pipeline error: {str(e)}")
            result['error'] = str(e)
            self.stats['failed_reports'] += 1
            return result

    def generate_report_batch(
        self,
        images: list,
        patient_ids: Optional[list] = None,
        ages: Optional[list] = None,
        genders: Optional[list] = None
    ) -> list:
        """
        Generate reports for multiple images

        Args:
            images: List of image paths or PIL Images
            patient_ids: List of patient IDs
            ages: List of ages
            genders: List of genders

        Returns:
            List of result dictionaries
        """
        results = []

        for i, image in enumerate(images):
            patient_id = patient_ids[i] if patient_ids and i < len(patient_ids) else None
            age = ages[i] if ages and i < len(ages) else None
            gender = genders[i] if genders and i < len(genders) else None

            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing image {i+1}/{len(images)}")
            logger.info(f"{'=' * 70}")

            result = self.generate_report(
                image=image,
                patient_id=patient_id,
                age=age,
                gender=gender
            )

            results.append(result)

        return results

    def save_report(
        self,
        report_result: Dict[str, Any],
        output_dir: str = "data/reports/generated"
    ) -> Optional[str]:
        """
        Save a generated report to file

        Args:
            report_result: Result dictionary from generate_report()
            output_dir: Directory to save reports

        Returns:
            Path to saved file or None if failed
        """
        if not report_result['success']:
            logger.error("Cannot save failed report")
            return None

        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            patient_id = report_result.get('patient_id', 'UNKNOWN')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"report_{patient_id}_{timestamp}.txt"
            filepath = output_path / filename

            # Write report
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("RADIOLOGY REPORT\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"Patient ID: {report_result.get('patient_id', 'N/A')}\n")
                f.write(f"Date: {report_result['timestamp']}\n")
                f.write(f"Image: {Path(report_result['image_path']).name}\n")
                f.write("\n" + "=" * 70 + "\n\n")

                f.write(report_result['report_text'])

                f.write("\n\n" + "=" * 70 + "\n")
                f.write("METADATA\n")
                f.write("=" * 70 + "\n")
                f.write(f"Total time: {report_result['total_time']:.2f}s\n")
                f.write(f"Vision time: {report_result['vision_time']:.2f}s\n")
                f.write(f"RAG time: {report_result.get('rag_time', 0):.2f}s\n")
                f.write(f"LLM time: {report_result['llm_time']:.2f}s\n")
                f.write(f"Tokens used: {report_result.get('tokens_used', 'N/A')}\n")
                f.write(f"\nGenerated by MedAssist Copilot\n")

            logger.info(f"✅ Report saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.stats,
            'vision_stats': self.vision_analyzer.get_stats() if self.vision_analyzer else {},
            'llm_stats': self.llm_processor.get_stats() if self.llm_processor else {}
        }

    def print_report(self, report_result: Dict[str, Any]):
        """Pretty print a generated report"""
        if not report_result['success']:
            print("\n❌ Report generation failed")
            print(f"Error: {report_result.get('error', 'Unknown error')}")
            return

        print("\n" + "=" * 70)
        print("GENERATED RADIOLOGY REPORT")
        print("=" * 70)
        print(f"\nPatient ID: {report_result.get('patient_id', 'N/A')}")
        print(f"Date: {report_result['timestamp']}")
        print(f"Image: {Path(report_result['image_path']).name}")
        print("\n" + "─" * 70 + "\n")

        print(report_result['report_text'])

        print("\n" + "─" * 70)
        print(f"\n⏱️  Performance:")
        print(f"   Total time: {report_result['total_time']:.2f}s")
        print(f"   Vision analysis: {report_result['vision_time']:.2f}s")
        print(f"   RAG retrieval: {report_result.get('rag_time', 0):.2f}s")
        print(f"   LLM generation: {report_result['llm_time']:.2f}s")
        print(f"   Tokens used: {report_result.get('tokens_used', 'N/A')}")
        print("=" * 70 + "\n")


# Main execution for testing
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "REPORT GENERATION PIPELINE TEST")
    print("=" * 70)

    # Check API key
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found!")
        print("   Please set OPENAI_API_KEY in your .env file")
        exit(1)

    # Find test image
    data_dir = Path("data/raw")
    test_image = None

    # Try to find a NORMAL image first
    normal_dir = data_dir / "NORMAL"
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg"))
        if normal_images:
            test_image = str(normal_images[0])

    if not test_image:
        # Try PNEUMONIA
        pneumonia_dir = data_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))
            if pneumonia_images:
                test_image = str(pneumonia_images[0])

    if not test_image:
        print("\n❌ No test images found in data/raw/")
        exit(1)

    print(f"\n📁 Test image: {Path(test_image).name}")

    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    pipeline = ReportGenerationPipeline(use_rag=True)

    # Generate report
    print("\n🚀 Starting report generation...\n")
    result = pipeline.generate_report(
        image=test_image,
        patient_id="P001",
        age=65,
        gender="M"
    )

    # Display report
    pipeline.print_report(result)

    # Show statistics
    print("📊 Pipeline Statistics:")
    stats = pipeline.get_stats()
    print(f"   Total reports generated: {stats['total_reports']}")
    print(f"   Successful: {stats['successful_reports']}")
    print(f"   Failed: {stats['failed_reports']}")
    if stats['successful_reports'] > 0:
        print(f"   Average time: {stats['average_time']:.2f}s")

    print("\n" + "=" * 70)
    print("Pipeline test complete!")
    print("=" * 70 + "\n")
