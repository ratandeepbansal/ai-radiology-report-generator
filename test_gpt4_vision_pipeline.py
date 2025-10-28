"""
Test Script for GPT-4 Vision Medical Analysis Pipeline
Tests the enhanced vision analysis with medical-grade GPT-4 Vision

Author: MedAssist Copilot Team
Created: Week 4.5 - Medical Vision Upgrade
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.pipeline import ReportGenerationPipeline
from src.vision_gpt4 import GPT4VisionAnalyzer

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text: str):
    """Print formatted section"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def test_gpt4_vision_standalone():
    """Test GPT-4 Vision analyzer standalone"""
    print_header("TEST 1: GPT-4 Vision Analyzer (Standalone)")

    # Find test images
    data_dir = Path("data/raw")
    normal_dir = data_dir / "NORMAL"
    pneumonia_dir = data_dir / "PNEUMONIA"

    test_images = []
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.jpeg"))[:1]
        test_images.extend([(img, "NORMAL") for img in normal_images])

    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.jpeg"))[:1]
        test_images.extend([(img, "PNEUMONIA") for img in pneumonia_images])

    if not test_images:
        print("\n❌ No test images found!")
        return False

    print(f"\n📁 Found {len(test_images)} test images")

    # Initialize analyzer
    print("\n🔧 Initializing GPT-4 Vision Analyzer...")
    try:
        analyzer = GPT4VisionAnalyzer(model_name="gpt-4o")
        print("   ✅ Analyzer initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False

    # Test each image
    for image_path, label in test_images:
        print_section(f"Analyzing: {image_path.name} (Actual: {label})")

        try:
            result = analyzer.analyze_xray(image_path)

            print(f"\n📊 Analysis Results:")
            print(f"   Status: {'NORMAL' if result['is_normal'] else 'ABNORMAL'}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")

            if result['detected_pathologies']:
                print(f"\n🔍 Detected Pathologies ({len(result['detected_pathologies'])}):")
                for pathology in result['detected_pathologies']:
                    print(f"      • {pathology['pathology']}: {pathology['confidence']} confidence")
            else:
                print(f"\n✅ No significant pathologies detected")

            # Show summary
            summary = analyzer.generate_findings_summary(result)
            print(f"\n📝 Summary for LLM:")
            print(f"   {summary[:300]}...")

            # Verify correctness
            if label == "NORMAL" and result['is_normal']:
                print(f"\n✅ CORRECT: Detected as NORMAL")
            elif label == "PNEUMONIA" and not result['is_normal']:
                print(f"\n✅ CORRECT: Detected as ABNORMAL")
            else:
                print(f"\n⚠️  MISMATCH: Expected {label}, got {'NORMAL' if result['is_normal'] else 'ABNORMAL'}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False

    return True


def test_pipeline_with_gpt4():
    """Test full pipeline with GPT-4 Vision"""
    print_header("TEST 2: Complete Pipeline with GPT-4 Vision")

    # Find test image
    data_dir = Path("data/raw")
    pneumonia_dir = data_dir / "PNEUMONIA"

    if not pneumonia_dir.exists():
        print("\n❌ Pneumonia directory not found!")
        return False

    test_images = list(pneumonia_dir.glob("*.jpeg"))[:1]
    if not test_images:
        print("\n❌ No pneumonia images found!")
        return False

    image_path = test_images[0]
    print(f"\n📁 Test Image: {image_path.name}")

    # Initialize pipeline with GPT-4 Vision
    print("\n🔧 Initializing Pipeline with GPT-4 Vision backend...")
    try:
        pipeline = ReportGenerationPipeline(
            vision_backend="gpt4",
            use_rag=True
        )
        print("   ✅ Pipeline initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False

    # Generate report
    print_section("Generating Report")

    try:
        result = pipeline.generate_report(
            image=str(image_path),
            patient_id="TEST001",
            age=45,
            gender="M"
        )

        if not result['success']:
            print(f"\n❌ Report generation failed: {result.get('error', 'Unknown error')}")
            return False

        print("\n✅ Report Generated Successfully!")

        # Display timing statistics
        print(f"\n⏱️  Performance:")
        print(f"   Vision Analysis: {result['vision_time']:.2f}s")
        print(f"   RAG Retrieval: {result.get('rag_time', 0):.2f}s")
        print(f"   LLM Generation: {result['llm_time']:.2f}s")
        print(f"   Total Time: {result['total_time']:.2f}s")

        # Display vision details
        if 'vision_details' in result:
            details = result['vision_details']
            print(f"\n🔍 Vision Analysis Details:")
            print(f"   Backend: {result['vision_backend']}")
            print(f"   Status: {'NORMAL' if details.get('is_normal') else 'ABNORMAL'}")
            print(f"   Confidence: {details.get('confidence', 'N/A')}")

            pathologies = details.get('detected_pathologies', [])
            if pathologies:
                print(f"   Detected Pathologies:")
                for p in pathologies:
                    print(f"      • {p['pathology']}: {p['confidence']}")

        # Display report sections
        print_section("Generated Report")

        if 'report_sections' in result:
            sections = result['report_sections']

            if 'findings' in sections:
                print(f"\n📋 FINDINGS:")
                print(f"{sections['findings']}")

            if 'impression' in sections:
                print(f"\n💡 IMPRESSION:")
                print(f"{sections['impression']}")

            if 'recommendations' in sections:
                print(f"\n🎯 RECOMMENDATIONS:")
                print(f"{sections['recommendations']}")
        else:
            print(f"\n{result['report_text']}")

        # Token usage
        print(f"\n📊 LLM Metrics:")
        print(f"   Tokens Used: {result.get('tokens_used', 'N/A')}")

        return True

    except Exception as e:
        print(f"\n❌ Error during report generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_comparison():
    """Compare BLIP-2 vs GPT-4 Vision"""
    print_header("TEST 3: Backend Comparison (BLIP-2 vs GPT-4 Vision)")

    # Find test image
    data_dir = Path("data/raw")
    pneumonia_dir = data_dir / "PNEUMONIA"

    if not pneumonia_dir.exists():
        print("\n❌ Pneumonia directory not found!")
        return False

    test_images = list(pneumonia_dir.glob("*.jpeg"))[:1]
    if not test_images:
        print("\n❌ No pneumonia images found!")
        return False

    image_path = test_images[0]
    print(f"\n📁 Test Image: {image_path.name}")

    results = {}

    # Test both backends
    for backend in ['blip', 'gpt4']:
        print_section(f"Testing with {backend.upper()} backend")

        try:
            pipeline = ReportGenerationPipeline(
                vision_backend=backend,
                use_rag=False  # Disable RAG for fair comparison
            )

            start_time = time.time()
            result = pipeline.generate_report(
                image=str(image_path),
                patient_id="COMP001",
                age=50,
                gender="F"
            )
            total_time = time.time() - start_time

            if result['success']:
                results[backend] = {
                    'vision_time': result['vision_time'],
                    'llm_time': result['llm_time'],
                    'total_time': total_time,
                    'vision_caption': result['vision_caption'][:200],
                    'report_length': len(result['report_text'])
                }

                print(f"   ✅ {backend.upper()} completed")
                print(f"   Vision Time: {result['vision_time']:.2f}s")
                print(f"   Total Time: {total_time:.2f}s")
                print(f"   Caption: {result['vision_caption'][:150]}...")
            else:
                print(f"   ❌ {backend.upper()} failed: {result.get('error')}")

        except Exception as e:
            print(f"   ❌ Error with {backend}: {e}")

    # Comparison summary
    if len(results) == 2:
        print_section("Comparison Summary")

        print(f"\n⏱️  Performance:")
        print(f"   BLIP-2 Vision Time: {results['blip']['vision_time']:.2f}s")
        print(f"   GPT-4 Vision Time: {results['gpt4']['vision_time']:.2f}s")
        speedup = results['blip']['vision_time'] / results['gpt4']['vision_time']
        if speedup > 1:
            print(f"   ⚡ GPT-4 is {speedup:.1f}x FASTER")
        else:
            print(f"   🐢 BLIP-2 is {1/speedup:.1f}x faster")

        print(f"\n📝 Vision Output Quality:")
        print(f"\n   BLIP-2 Caption (generic):")
        print(f"   \"{results['blip']['vision_caption']}\"")
        print(f"\n   GPT-4 Caption (medical-specific):")
        print(f"   \"{results['gpt4']['vision_caption']}\"")

        print(f"\n🏆 Recommendation: GPT-4 Vision provides medically relevant findings")

    return len(results) == 2


def main():
    """Run all tests"""
    print_header("MedAssist Copilot - GPT-4 Vision Integration Tests")

    # Check API key
    if not config.OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key in .env file")
        return

    print(f"\n✅ Configuration:")
    print(f"   Vision Backend: {config.VISION_BACKEND}")
    print(f"   GPT-4 Vision Model: {config.GPT4_VISION_MODEL}")
    print(f"   LLM Model: {config.LLM_MODEL_NAME}")

    # Run tests
    tests = [
        ("GPT-4 Vision Standalone", test_gpt4_vision_standalone),
        ("Full Pipeline with GPT-4", test_pipeline_with_gpt4),
        ("Backend Comparison", test_pipeline_comparison),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print_header("Test Summary")
    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! GPT-4 Vision integration successful!")
        print("\n💡 Next steps:")
        print("   1. Update app.py UI to display pathology findings")
        print("   2. Test with larger dataset")
        print("   3. Compare report quality metrics (BLEU/ROUGE)")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")


if __name__ == "__main__":
    main()
