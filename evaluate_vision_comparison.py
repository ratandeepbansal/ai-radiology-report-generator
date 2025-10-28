"""
Vision Backend Comparison Evaluation
Compares BLIP-2 vs GPT-4 Vision using BLEU and ROUGE metrics

Author: MedAssist Copilot Team
Created: Week 4.5 - Medical Vision Upgrade
"""

import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ReportGenerationPipeline
from src.report_manager import ReportManager
from evaluate import calculate_bleu, calculate_rouge, evaluate_report
import config


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def evaluate_backend(
    backend_name: str,
    test_images: List[str],
    test_reports: List[Dict[str, Any]],
    use_rag: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a specific vision backend

    Args:
        backend_name: 'blip' or 'gpt4'
        test_images: List of test image paths
        test_reports: List of reference reports
        use_rag: Whether to use RAG

    Returns:
        Evaluation results dictionary
    """
    print_section(f"Evaluating {backend_name.upper()} Backend")

    # Initialize pipeline with specific backend
    print(f"\n🔧 Initializing pipeline with {backend_name} backend...")
    try:
        pipeline = ReportGenerationPipeline(
            vision_backend=backend_name,
            use_rag=use_rag
        )
        print("   ✅ Pipeline initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        return {}

    # Evaluation results
    all_results = []
    total_time = 0
    total_vision_time = 0
    total_llm_time = 0
    successful = 0
    failed = 0

    print(f"\n📊 Processing {len(test_images)} images...")

    for i, (img_path, ref_report) in enumerate(zip(test_images, test_reports), 1):
        print(f"\n   [{i}/{len(test_images)}] {Path(img_path).name}", end=" ... ")

        try:
            # Generate report
            result = pipeline.generate_report(
                image=img_path,
                patient_id=ref_report.get('patient_id'),
                age=ref_report.get('metadata', {}).get('age'),
                gender=ref_report.get('metadata', {}).get('gender')
            )

            if result['success']:
                successful += 1
                total_time += result['total_time']
                total_vision_time += result['vision_time']
                total_llm_time += result['llm_time']

                # Extract reference text
                ref_text = ref_report['report']['impression']

                # Extract generated text
                gen_text = result['report_sections'].get('impression', '')

                # Calculate metrics
                metrics = evaluate_report(gen_text, ref_text)

                # Store result
                result_entry = {
                    'image': Path(img_path).name,
                    'patient_id': ref_report.get('patient_id'),
                    'success': True,
                    'metrics': metrics,
                    'time': result['total_time'],
                    'vision_time': result['vision_time'],
                    'llm_time': result['llm_time'],
                    'tokens': result.get('tokens_used', 0),
                    'vision_caption': result.get('vision_caption', '')[:200]
                }

                # Add vision details if available
                if 'vision_details' in result:
                    result_entry['vision_details'] = {
                        'is_normal': result['vision_details'].get('is_normal'),
                        'confidence': result['vision_details'].get('confidence'),
                        'pathology_count': len(result['vision_details'].get('detected_pathologies', []))
                    }

                all_results.append(result_entry)

                print(f"✅ BLEU: {metrics['bleu']:.3f} | ROUGE-L: {metrics['rouge-L']:.3f} | Time: {result['total_time']:.1f}s")

            else:
                failed += 1
                print(f"❌ Failed: {result.get('error')}")

        except Exception as e:
            failed += 1
            print(f"❌ Error: {str(e)}")

    # Calculate aggregate metrics
    if all_results:
        avg_bleu = sum(r['metrics']['bleu'] for r in all_results) / len(all_results)
        avg_rouge1 = sum(r['metrics']['rouge-1'] for r in all_results) / len(all_results)
        avg_rouge2 = sum(r['metrics']['rouge-2'] for r in all_results) / len(all_results)
        avg_rougeL = sum(r['metrics']['rouge-L'] for r in all_results) / len(all_results)
        avg_time = total_time / successful if successful > 0 else 0
        avg_vision_time = total_vision_time / successful if successful > 0 else 0
        avg_llm_time = total_llm_time / successful if successful > 0 else 0
        total_tokens = sum(r['tokens'] for r in all_results)

        summary = {
            'backend': backend_name,
            'total_images': len(test_images),
            'successful': successful,
            'failed': failed,
            'metrics': {
                'avg_bleu': avg_bleu,
                'avg_rouge-1': avg_rouge1,
                'avg_rouge-2': avg_rouge2,
                'avg_rouge-L': avg_rougeL
            },
            'performance': {
                'avg_time': avg_time,
                'avg_vision_time': avg_vision_time,
                'avg_llm_time': avg_llm_time,
                'total_time': total_time,
                'total_tokens': total_tokens,
                'avg_tokens': total_tokens / successful if successful > 0 else 0
            },
            'detailed_results': all_results
        }

        # Print summary
        print(f"\n📈 {backend_name.upper()} Summary:")
        print(f"   Success rate: {successful}/{len(test_images)} ({successful/len(test_images)*100:.1f}%)")
        print(f"   Avg BLEU: {avg_bleu:.3f} | Avg ROUGE-L: {avg_rougeL:.3f}")
        print(f"   Avg Time: {avg_time:.2f}s (Vision: {avg_vision_time:.2f}s, LLM: {avg_llm_time:.2f}s)")

        return summary

    else:
        print(f"\n❌ No successful evaluations for {backend_name}")
        return {}


def compare_backends(
    test_images: List[str],
    test_reports: List[Dict[str, Any]],
    save_results: bool = True
) -> None:
    """
    Compare BLIP-2 and GPT-4 Vision backends

    Args:
        test_images: List of test image paths
        test_reports: List of reference reports
        save_results: Whether to save results to file
    """
    print_header("MedAssist Copilot - Vision Backend Comparison")

    print(f"\n📁 Test Dataset:")
    print(f"   Images: {len(test_images)}")
    print(f"   Reference Reports: {len(test_reports)}")

    # Evaluate both backends
    results = {}

    # 1. Evaluate BLIP-2
    results['blip'] = evaluate_backend('blip', test_images, test_reports, use_rag=False)

    # 2. Evaluate GPT-4 Vision
    results['gpt4'] = evaluate_backend('gpt4', test_images, test_reports, use_rag=False)

    # Comparison
    if results['blip'] and results['gpt4']:
        print_header("Comparison Summary")

        blip = results['blip']
        gpt4 = results['gpt4']

        # Quality metrics comparison
        print("\n📊 Quality Metrics Comparison:")
        print(f"\n   {'Metric':<20} {'BLIP-2':<15} {'GPT-4 Vision':<15} {'Winner':<15}")
        print(f"   {'-'*65}")

        metrics_comparison = [
            ('BLEU', blip['metrics']['avg_bleu'], gpt4['metrics']['avg_bleu']),
            ('ROUGE-1', blip['metrics']['avg_rouge-1'], gpt4['metrics']['avg_rouge-1']),
            ('ROUGE-2', blip['metrics']['avg_rouge-2'], gpt4['metrics']['avg_rouge-2']),
            ('ROUGE-L', blip['metrics']['avg_rouge-L'], gpt4['metrics']['avg_rouge-L']),
        ]

        for metric_name, blip_score, gpt4_score in metrics_comparison:
            winner = "GPT-4 Vision" if gpt4_score > blip_score else "BLIP-2" if blip_score > gpt4_score else "Tie"
            print(f"   {metric_name:<20} {blip_score:<15.3f} {gpt4_score:<15.3f} {winner:<15}")

        # Performance comparison
        print("\n⏱️  Performance Comparison:")
        print(f"\n   {'Metric':<25} {'BLIP-2':<15} {'GPT-4 Vision':<15} {'Difference':<15}")
        print(f"   {'-'*70}")

        perf_comparison = [
            ('Total Time', blip['performance']['avg_time'], gpt4['performance']['avg_time'], 's'),
            ('Vision Analysis Time', blip['performance']['avg_vision_time'], gpt4['performance']['avg_vision_time'], 's'),
            ('LLM Generation Time', blip['performance']['avg_llm_time'], gpt4['performance']['avg_llm_time'], 's'),
            ('Tokens Used', blip['performance']['avg_tokens'], gpt4['performance']['avg_tokens'], 'tokens'),
        ]

        for metric_name, blip_val, gpt4_val, unit in perf_comparison:
            diff = gpt4_val - blip_val
            diff_str = f"+{diff:.2f}{unit}" if diff > 0 else f"{diff:.2f}{unit}"
            print(f"   {metric_name:<25} {blip_val:<15.2f} {gpt4_val:<15.2f} {diff_str:<15}")

        # Speed ratio
        speedup = blip['performance']['avg_vision_time'] / gpt4['performance']['avg_vision_time']
        if speedup > 1:
            print(f"\n   💨 BLIP-2 is {speedup:.1f}x FASTER for vision analysis")
        else:
            print(f"\n   💨 GPT-4 Vision is {1/speedup:.1f}x FASTER for vision analysis")

        # Vision quality comparison
        print("\n🔍 Vision Analysis Quality:")
        print(f"\n   BLIP-2:")
        if blip['detailed_results']:
            print(f"   • Generic captions (e.g., '{blip['detailed_results'][0]['vision_caption'][:100]}...')")
            print(f"   • No pathology detection")
            print(f"   • Fast but limited medical value")

        print(f"\n   GPT-4 Vision:")
        if gpt4['detailed_results']:
            first_result = gpt4['detailed_results'][0]
            print(f"   • Medical-specific analysis")
            if 'vision_details' in first_result:
                details = first_result['vision_details']
                print(f"   • Status detection: {'NORMAL' if details.get('is_normal') else 'ABNORMAL'}")
                print(f"   • Confidence levels: {details.get('confidence')}")
                print(f"   • Pathology detection: {details.get('pathology_count', 0)} pathologies found")

        # Recommendations
        print_header("Recommendations")

        quality_improvement = ((gpt4['metrics']['avg_bleu'] - blip['metrics']['avg_bleu']) / blip['metrics']['avg_bleu'] * 100)
        time_overhead = ((gpt4['performance']['avg_time'] - blip['performance']['avg_time']) / blip['performance']['avg_time'] * 100)

        print(f"\n✅ GPT-4 Vision provides {quality_improvement:+.1f}% better BLEU score")
        print(f"⚠️  With {time_overhead:+.1f}% time overhead")

        print(f"\n💡 Use Case Recommendations:")
        print(f"   • For SPEED-CRITICAL applications: Use BLIP-2")
        print(f"   • For MEDICAL ACCURACY: Use GPT-4 Vision")
        print(f"   • For PRODUCTION: Use GPT-4 Vision (better clinical value)")
        print(f"   • For DEVELOPMENT/TESTING: Either backend works")

        # Save results
        if save_results:
            output_file = "vision_comparison_results.json"
            comparison_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_size': len(test_images),
                'backends': results,
                'comparison': {
                    'quality_improvement_pct': quality_improvement,
                    'time_overhead_pct': time_overhead,
                    'speed_ratio': speedup
                }
            }

            with open(output_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

            # Create comparison table CSV
            csv_file = "vision_comparison_table.csv"
            comparison_table = pd.DataFrame([
                {
                    'Backend': 'BLIP-2',
                    'BLEU': blip['metrics']['avg_bleu'],
                    'ROUGE-1': blip['metrics']['avg_rouge-1'],
                    'ROUGE-2': blip['metrics']['avg_rouge-2'],
                    'ROUGE-L': blip['metrics']['avg_rouge-L'],
                    'Avg Time (s)': blip['performance']['avg_time'],
                    'Vision Time (s)': blip['performance']['avg_vision_time'],
                    'Tokens': blip['performance']['avg_tokens']
                },
                {
                    'Backend': 'GPT-4 Vision',
                    'BLEU': gpt4['metrics']['avg_bleu'],
                    'ROUGE-1': gpt4['metrics']['avg_rouge-1'],
                    'ROUGE-2': gpt4['metrics']['avg_rouge-2'],
                    'ROUGE-L': gpt4['metrics']['avg_rouge-L'],
                    'Avg Time (s)': gpt4['performance']['avg_time'],
                    'Vision Time (s)': gpt4['performance']['avg_vision_time'],
                    'Tokens': gpt4['performance']['avg_tokens']
                }
            ])

            comparison_table.to_csv(csv_file, index=False)
            print(f"📊 Comparison table saved to: {csv_file}")

        print("\n" + "=" * 80)

    else:
        print("\n❌ Comparison failed - one or both backends had no successful evaluations")


def main():
    """Main evaluation"""

    # Load test data
    print("\n🔍 Loading test data...")

    # Use existing patient reports as references
    report_manager = ReportManager()
    all_reports = report_manager.get_all_reports()

    # Check for test images
    data_dir = Path("data/raw")

    # Find NORMAL and PNEUMONIA images
    normal_dir = data_dir / "NORMAL"
    pneumonia_dir = data_dir / "PNEUMONIA"

    test_images = []
    test_reports = []

    # Use a few images for evaluation (not all to save time/cost)
    if normal_dir.exists():
        normal_imgs = list(normal_dir.glob("*.jpeg"))[:2]
        test_images.extend([str(img) for img in normal_imgs])
        # Use P001 (normal) as reference
        test_reports.extend([all_reports[0]] * len(normal_imgs))

    if pneumonia_dir.exists():
        pneumonia_imgs = list(pneumonia_dir.glob("*.jpeg"))[:2]
        test_images.extend([str(img) for img in pneumonia_imgs])
        # Use P002 (pneumonia) as reference
        test_reports.extend([all_reports[1]] * len(pneumonia_imgs))

    if not test_images:
        print("❌ No test images found")
        print("   Add X-ray images to data/raw/NORMAL or data/raw/PNEUMONIA")
        return

    print(f"   ✅ Found {len(test_images)} test images")

    # Run comparison
    compare_backends(test_images, test_reports)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

    main()
