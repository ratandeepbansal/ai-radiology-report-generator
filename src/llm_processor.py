"""
LLM Processor Module for MedAssist Copilot
Handles integration with OpenAI GPT models for report generation
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import time
import json

from openai import OpenAI
from openai import OpenAIError, RateLimitError, APIError

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProcessor:
    """
    LLM processor for generating radiology reports
    Uses OpenAI's GPT models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize the LLM processor

        Args:
            api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
            model_name: Model name (defaults to config.LLM_MODEL_NAME)
            max_tokens: Maximum tokens for generation (defaults to config.LLM_MAX_TOKENS)
            temperature: Temperature for generation (defaults to config.LLM_TEMPERATURE)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.temperature = temperature or config.LLM_TEMPERATURE

        # Validate API key
        if not self.api_key:
            logger.error(config.ERROR_MESSAGES['no_api_key'])
            raise ValueError("OpenAI API key is required")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"LLM Processor initialized with model: {self.model_name}")

        # Statistics
        self.stats = {
            'reports_generated': 0,
            'total_tokens_used': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'errors': 0
        }

    def generate_report(
        self,
        vision_caption: str,
        patient_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        rag_context: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a radiology report using the LLM

        Args:
            vision_caption: Caption/description from vision model
            patient_id: Patient identifier
            age: Patient age
            gender: Patient gender
            rag_context: Context from prior reports (RAG)
            prompt_template: Custom prompt template
            max_retries: Maximum number of API call retries

        Returns:
            Dictionary with report and metadata, or None if failed
        """
        # Use default prompt template if not provided
        if prompt_template is None:
            prompt_template = config.REPORT_GENERATION_PROMPT

        # Fill in the prompt template
        prompt = prompt_template.format(
            vision_caption=vision_caption,
            rag_context=rag_context or "No prior reports available for this patient.",
            patient_id=patient_id or "Unknown",
            age=age or "Unknown",
            gender=gender or "Unknown",
            date=time.strftime("%Y-%m-%d")
        )

        # Try to generate report with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating report (attempt {attempt + 1}/{max_retries})...")
                start_time = time.time()

                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": config.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=config.LLM_TOP_P
                )

                # Extract response
                report_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                generation_time = time.time() - start_time

                # Update statistics
                self.stats['reports_generated'] += 1
                self.stats['total_tokens_used'] += tokens_used
                self.stats['total_time'] += generation_time
                self.stats['average_time'] = self.stats['total_time'] / self.stats['reports_generated']

                logger.info(f"✅ Report generated in {generation_time:.2f}s")
                logger.info(f"   Tokens used: {tokens_used}")

                # Parse report into sections
                report_sections = self._parse_report(report_text)

                # Return structured result
                return {
                    'success': True,
                    'report_text': report_text,
                    'sections': report_sections,
                    'metadata': {
                        'patient_id': patient_id,
                        'age': age,
                        'gender': gender,
                        'date': time.strftime("%Y-%m-%d"),
                        'model': self.model_name,
                        'tokens_used': tokens_used,
                        'generation_time': generation_time
                    }
                }

            except RateLimitError as e:
                logger.warning(f"Rate limit hit: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    self.stats['errors'] += 1
                    logger.error("Max retries reached due to rate limit")
                    return None

            except APIError as e:
                logger.error(f"API error: {str(e)}")
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                self.stats['errors'] += 1
                return None

        return None

    def _parse_report(self, report_text: str) -> Dict[str, str]:
        """
        Parse report text into structured sections

        Args:
            report_text: Raw report text from LLM

        Returns:
            Dictionary with parsed sections
        """
        sections = {
            'findings': '',
            'impression': '',
            'recommendations': ''
        }

        # Split by section headers (case-insensitive)
        text = report_text

        # Extract FINDINGS
        if '**FINDINGS:**' in text or '**Findings:**' in text:
            findings_start = text.find('**FINDINGS:**')
            if findings_start == -1:
                findings_start = text.find('**Findings:**')
            findings_start += len('**FINDINGS:**')

            findings_end = text.find('**IMPRESSION:**', findings_start)
            if findings_end == -1:
                findings_end = text.find('**Impression:**', findings_start)

            if findings_end != -1:
                sections['findings'] = text[findings_start:findings_end].strip()
            else:
                sections['findings'] = text[findings_start:].strip()

        # Extract IMPRESSION
        if '**IMPRESSION:**' in text or '**Impression:**' in text:
            impression_start = text.find('**IMPRESSION:**')
            if impression_start == -1:
                impression_start = text.find('**Impression:**')
            impression_start += len('**IMPRESSION:**')

            impression_end = text.find('**RECOMMENDATIONS:**', impression_start)
            if impression_end == -1:
                impression_end = text.find('**Recommendations:**', impression_start)

            if impression_end != -1:
                sections['impression'] = text[impression_start:impression_end].strip()
            else:
                sections['impression'] = text[impression_start:].strip()

        # Extract RECOMMENDATIONS
        if '**RECOMMENDATIONS:**' in text or '**Recommendations:**' in text:
            recommendations_start = text.find('**RECOMMENDATIONS:**')
            if recommendations_start == -1:
                recommendations_start = text.find('**Recommendations:**')
            recommendations_start += len('**RECOMMENDATIONS:**')

            sections['recommendations'] = text[recommendations_start:].strip()

        return sections

    def generate_comparison_report(
        self,
        current_caption: str,
        previous_caption: str,
        previous_report: Optional[str] = None,
        patient_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a comparison report between current and previous X-rays

        Args:
            current_caption: Caption for current X-ray
            previous_caption: Caption for previous X-ray
            previous_report: Previous radiology report text
            patient_id: Patient identifier

        Returns:
            Dictionary with comparison report
        """
        prompt = config.COMPARISON_PROMPT.format(
            current_findings=current_caption,
            previous_findings=previous_caption,
            previous_report=previous_report or "Not available",
            patient_id=patient_id or "Unknown",
            current_date=time.strftime("%Y-%m-%d"),
            previous_date="Previous scan"
        )

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            report_text = response.choices[0].message.content
            generation_time = time.time() - start_time

            logger.info(f"✅ Comparison report generated in {generation_time:.2f}s")

            return {
                'success': True,
                'report_text': report_text,
                'generation_time': generation_time
            }

        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return None

    def test_api_connection(self) -> bool:
        """
        Test the API connection with a simple request

        Returns:
            True if connection works, False otherwise
        """
        try:
            logger.info("Testing API connection...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Respond with 'OK' if you can read this."}
                ],
                max_tokens=10
            )

            result = response.choices[0].message.content
            logger.info(f"✅ API connection successful. Response: {result}")
            return True

        except Exception as e:
            logger.error(f"❌ API connection failed: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'reports_generated': 0,
            'total_tokens_used': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'errors': 0
        }
        logger.info("Statistics reset")


# Utility function
def quick_report(
    vision_caption: str,
    patient_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Quick function to generate a report from a caption

    Args:
        vision_caption: Caption from vision model
        patient_id: Patient identifier
        api_key: Optional API key

    Returns:
        Generated report text or None
    """
    try:
        processor = LLMProcessor(api_key=api_key)
        result = processor.generate_report(
            vision_caption=vision_caption,
            patient_id=patient_id
        )
        return result['report_text'] if result else None
    except Exception as e:
        logger.error(f"Error in quick_report: {str(e)}")
        return None


# Main execution for testing
if __name__ == "__main__":
    print("=" * 70)
    print("MedAssist Copilot - LLM Processor Test")
    print("=" * 70)

    # Check API key
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found!")
        print("   Please set OPENAI_API_KEY in your .env file")
        print("   1. Copy .env.example to .env")
        print("   2. Add your OpenAI API key")
        print("   3. Run this script again")
        exit(1)

    # Initialize processor
    print("\n1️⃣  Initializing LLM Processor...")
    try:
        processor = LLMProcessor()
        print(f"   ✅ Initialized with model: {processor.model_name}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        exit(1)

    # Test API connection
    print("\n2️⃣  Testing API connection...")
    if not processor.test_api_connection():
        print("   ❌ API connection test failed")
        exit(1)

    # Test report generation with sample caption
    print("\n3️⃣  Testing report generation...")
    sample_caption = (
        "A chest X-ray image showing bilateral lung fields with increased "
        "opacity in the right lower lobe, suggesting possible pneumonia. "
        "The cardiac silhouette appears normal in size."
    )

    print(f"   Sample caption: {sample_caption[:80]}...")

    result = processor.generate_report(
        vision_caption=sample_caption,
        patient_id="TEST001",
        age=65,
        gender="M",
        rag_context="No prior reports available."
    )

    if result and result['success']:
        print("\n   ✅ Report generated successfully!")
        print("\n   " + "─" * 66)
        print("   GENERATED REPORT:")
        print("   " + "─" * 66)

        # Print the report
        lines = result['report_text'].split('\n')
        for line in lines:
            print(f"   {line}")

        print("   " + "─" * 66)

        # Print metadata
        print(f"\n   📊 Metadata:")
        print(f"      • Tokens used: {result['metadata']['tokens_used']}")
        print(f"      • Generation time: {result['metadata']['generation_time']:.2f}s")
        print(f"      • Model: {result['metadata']['model']}")

    else:
        print("   ❌ Failed to generate report")

    # Show statistics
    print("\n4️⃣  Processing Statistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("LLM Processor test complete!")
    print("=" * 70)
