"""
Week 2 Pipeline Test
Comprehensive test of Vision-Language pipeline and report generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vision import VisionAnalyzer
from src.llm_processor import LLMProcessor
from src.pipeline import ReportGenerationPipeline
import config


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_vision_module():
    """Test the vision analysis module"""
    print_section("VISION MODULE TEST")

    print("\n1️⃣  Initializing Vision Analyzer...")
    analyzer = VisionAnalyzer()

    # Find test images
    data_dir = Path("data/raw")
    test_images = []

    # Get NORMAL and PNEUMONIA samples
    for category in ["NORMAL", "PNEUMONIA"]:
        cat_dir = data_dir / category
        if cat_dir.exists():
            images = list(cat_dir.glob("*.jpeg"))[:2]
            test_images.extend(images)

    if not test_images:
        print("   ⚠️  No test images found")
        return False

    print(f"   ✅ Found {len(test_images)} test images")

    # Load model
    print("\n2️⃣  Loading BLIP vision model...")
    if not analyzer.load_model():
        print("   ❌ Failed to load model")
        return False

    # Test captions
    print("\n3️⃣  Generating image captions...")
    for i, img_path in enumerate(test_images[:2], 1):  # Test first 2
        print(f"\n   Image {i}: {img_path.name}")
        caption = analyzer.generate_caption(str(img_path))

        if caption:
            print(f"   ✅ Caption: {caption}")
        else:
            print("   ❌ Failed")
            return False

    # Test medical description
    print("\n4️⃣  Testing medical description generation...")
    test_img = test_images[0]
    print(f"   Image: {test_img.name}")
    medical_desc = analyzer.generate_medical_description(str(test_img))

    if medical_desc:
        print(f"   ✅ Description: {medical_desc}")
    else:
        print("   ❌ Failed")

    # Statistics
    print("\n5️⃣  Vision Module Statistics:")
    stats = analyzer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    return True


def test_llm_module():
    """Test the LLM processor module"""
    print_section("LLM PROCESSOR TEST")

    # Check API key
    if not config.OPENAI_API_KEY:
        print("\n❌ OpenAI API key not found!")
        print("   To test LLM functionality:")
        print("   1. Get API key from https://platform.openai.com/api-keys")
        print("   2. Copy .env.example to .env")
        print("   3. Add your API key to .env")
        print("   4. Run this test again")
        return False

    print("\n1️⃣  Initializing LLM Processor...")
    try:
        processor = LLMProcessor()
        print(f"   ✅ Initialized with model: {processor.model_name}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return False

    # Test API connection
    print("\n2️⃣  Testing API connection...")
    if not processor.test_api_connection():
        print("   ❌ API connection failed")
        return False

    # Test report generation
    print("\n3️⃣  Testing report generation...")
    sample_caption = (
        "A chest X-ray showing clear bilateral lung fields with normal "
        "cardiac silhouette and no signs of consolidation or effusion."
    )

    print(f"   Sample caption: {sample_caption[:60]}...")

    result = processor.generate_report(
        vision_caption=sample_caption,
        patient_id="TEST001",
        age=55,
        gender="F"
    )

    if result and result['success']:
        print("   ✅ Report generated successfully!")
        print(f"\n   {'─' * 66}")
        print(f"   SAMPLE REPORT (first 300 chars):")
        print(f"   {'─' * 66}")
        report_preview = result['report_text'][:300] + "..."
        for line in report_preview.split('\n'):
            print(f"   {line}")
        print(f"   {'─' * 66}")

        print(f"\n   📊 Tokens used: {result['metadata']['tokens_used']}")
        print(f"   ⏱️  Generation time: {result['metadata']['generation_time']:.2f}s")
    else:
        print("   ❌ Failed to generate report")
        return False

    # Statistics
    print("\n4️⃣  LLM Processor Statistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    return True


def test_complete_pipeline():
    """Test the complete end-to-end pipeline"""
    print_section("COMPLETE PIPELINE TEST")

    # Check API key
    if not config.OPENAI_API_KEY:
        print("\n⚠️  Skipping pipeline test (API key required)")
        return False

    # Find test image
    data_dir = Path("data/raw")
    test_image = None

    # Try NORMAL first
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
        print("   ❌ No test images found")
        return False

    print(f"\n📁 Test image: {Path(test_image).name}")

    # Initialize pipeline
    print("\n1️⃣  Initializing complete pipeline...")
    try:
        pipeline = ReportGenerationPipeline(use_rag=True)
        print("   ✅ Pipeline initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        return False

    # Generate report
    print("\n2️⃣  Running complete pipeline: Image → Vision → RAG → LLM...")
    print("   (This may take 1-2 minutes for the first run)")

    result = pipeline.generate_report(
        image=test_image,
        patient_id="P001",  # This patient exists in our sample reports
        age=65,
        gender="M"
    )

    # Display results
    print("\n3️⃣  Pipeline Results:")

    if result['success']:
        print("   ✅ Pipeline completed successfully!")

        print(f"\n   {'─' * 66}")
        print("   GENERATED REPORT:")
        print(f"   {'─' * 66}\n")

        # Print report (truncated)
        lines = result['report_text'].split('\n')
        for line in lines[:15]:  # First 15 lines
            print(f"   {line}")
        if len(lines) > 15:
            print(f"   ... ({len(lines) - 15} more lines)")

        print(f"\n   {'─' * 66}")

        # Performance metrics
        print(f"\n   ⏱️  Performance Breakdown:")
        print(f"      Total time:    {result['total_time']:.2f}s")
        print(f"      Vision:        {result['vision_time']:.2f}s ({result['vision_time']/result['total_time']*100:.1f}%)")
        print(f"      RAG:           {result.get('rag_time', 0):.2f}s ({result.get('rag_time', 0)/result['total_time']*100:.1f}%)")
        print(f"      LLM:           {result['llm_time']:.2f}s ({result['llm_time']/result['total_time']*100:.1f}%)")
        print(f"      Tokens used:   {result.get('tokens_used', 'N/A')}")

        # RAG info
        if result.get('prior_reports_count', 0) > 0:
            print(f"\n   📚 RAG Context: Retrieved {result['prior_reports_count']} prior report(s)")
        else:
            print(f"\n   📚 RAG Context: No prior reports found")

        # Save report
        print("\n4️⃣  Saving generated report...")
        saved_path = pipeline.save_report(result)
        if saved_path:
            print(f"   ✅ Report saved to: {saved_path}")
        else:
            print("   ⚠️  Could not save report")

    else:
        print(f"   ❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        return False

    # Pipeline statistics
    print("\n5️⃣  Pipeline Statistics:")
    stats = pipeline.get_stats()
    print(f"   Total reports generated: {stats['total_reports']}")
    print(f"   Successful: {stats['successful_reports']}")
    print(f"   Failed: {stats['failed_reports']}")
    if stats['successful_reports'] > 0:
        print(f"   Average time: {stats['average_time']:.2f}s")

    return True


def main():
    """Run all Week 2 tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "WEEK 2 PIPELINE - COMPREHENSIVE TEST")
    print("=" * 70)

    results = {}

    # Test 1: Vision Module
    try:
        results['vision'] = test_vision_module()
    except Exception as e:
        print(f"\n❌ Vision test error: {str(e)}")
        results['vision'] = False

    # Test 2: LLM Module
    try:
        results['llm'] = test_llm_module()
    except Exception as e:
        print(f"\n❌ LLM test error: {str(e)}")
        results['llm'] = False

    # Test 3: Complete Pipeline
    try:
        results['pipeline'] = test_complete_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline test error: {str(e)}")
        results['pipeline'] = False

    # Summary
    print_section("SUMMARY")

    print("\n📋 Test Results:")
    print(f"   Vision Module:      {'✅ PASSED' if results.get('vision') else '❌ FAILED'}")
    print(f"   LLM Processor:      {'✅ PASSED' if results.get('llm') else '❌ FAILED / SKIPPED'}")
    print(f"   Complete Pipeline:  {'✅ PASSED' if results.get('pipeline') else '❌ FAILED / SKIPPED'}")

    # Check if all passed
    if all(results.values()):
        print("\n✅ All Week 2 components are working correctly!")

        print("\n📋 Week 2 Deliverables:")
        print("   ✅ Vision-to-text pipeline (BLIP-2)")
        print("   ✅ LLM report generation (GPT)")
        print("   ✅ End-to-end integration")
        print("   ✅ RAG integration with prior reports")
        print("   ✅ Performance metrics and logging")

        print("\n🎯 Ready for Week 3 Tasks:")
        print("   - RAG system enhancement (vector database)")
        print("   - Voice input integration (Whisper)")
        print("   - Voice dictation for report editing")

    else:
        print("\n⚠️  Some tests failed or were skipped")
        if not results.get('llm') or not results.get('pipeline'):
            print("\n💡 To enable full testing:")
            print("   1. Get OpenAI API key: https://platform.openai.com/api-keys")
            print("   2. Copy .env.example to .env")
            print("   3. Add your API key to .env")
            print("   4. Run this test again")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
