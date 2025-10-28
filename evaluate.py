"""
Evaluation Script for MedAssist Copilot
Measures report quality using BLEU, ROUGE, and other metrics
"""

import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ReportGenerationPipeline
from src.report_manager import ReportManager
import config


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score

    Args:
        reference: Reference (ground truth) text
        hypothesis: Generated text

    Returns:
        BLEU score (0-1)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        # Tokenize
        reference_tokens = [word_tokenize(reference.lower())]
        hypothesis_tokens = word_tokenize(hypothesis.lower())

        # Calculate BLEU with smoothing
        smoothie = SmoothingFunction().method4
        score = sentence_bleu(
            reference_tokens,
            hypothesis_tokens,
            smoothing_function=smoothie
        )

        return score

    except ImportError:
        print("NLTK not installed. Install with: pip install nltk")
        return 0.0
    except Exception as e:
        print(f"Error calculating BLEU: {str(e)}")
        return 0.0


def calculate_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores

    Args:
        reference: Reference (ground truth) text
        hypothesis: Generated text

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        scores = scorer.score(reference, hypothesis)

        return {
            'rouge-1': scores['rouge1'].fmeasure,
            'rouge-2': scores['rouge2'].fmeasure,
            'rouge-L': scores['rougeL'].fmeasure
        }

    except ImportError:
        print("rouge-score not installed. Install with: pip install rouge-score")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-L': 0.0}
    except Exception as e:
        print(f"Error calculating ROUGE: {str(e)}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-L': 0.0}


def evaluate_report(
    generated_report: str,
    reference_report: str
) -> Dict[str, Any]:
    """
    Evaluate a generated report against a reference

    Args:
        generated_report: Generated report text
        reference_report: Reference (ground truth) report text

    Returns:
        Dictionary with evaluation metrics
    """
    results = {}

    # BLEU score
    bleu = calculate_bleu(reference_report, generated_report)
    results['bleu'] = bleu

    # ROUGE scores
    rouge = calculate_rouge(reference_report, generated_report)
    results.update(rouge)

    # Length metrics
    results['gen_length'] = len(generated_report)
    results['ref_length'] = len(reference_report)
    results['length_ratio'] = len(generated_report) / len(reference_report) if len(reference_report) > 0 else 0

    return results


def evaluate_pipeline(
    test_images: List[str],
    test_reports: List[Dict[str, Any]],
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the complete pipeline on a test set

    Args:
        test_images: List of test image paths
        test_reports: List of reference reports
        save_results: Whether to save results to file

    Returns:
        Evaluation results dictionary
    """
    print("=" * 70)
    print("MedAssist Copilot - Pipeline Evaluation")
    print("=" * 70)

    # Initialize pipeline
    print("\n1️⃣  Initializing pipeline...")
    try:
        pipeline = ReportGenerationPipeline(use_rag=True)
        print("   ✅ Pipeline initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        return {}

    # Evaluation results
    all_results = []
    total_time = 0
    successful = 0
    failed = 0

    print(f"\n2️⃣  Evaluating {len(test_images)} images...")

    for i, (img_path, ref_report) in enumerate(zip(test_images, test_reports), 1):
        print(f"\n   Processing {i}/{len(test_images)}: {Path(img_path).name}")

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
                    'tokens': result.get('tokens_used', 0)
                }

                all_results.append(result_entry)

                print(f"   ✅ BLEU: {metrics['bleu']:.3f} | ROUGE-L: {metrics['rouge-L']:.3f}")

            else:
                failed += 1
                print(f"   ❌ Failed: {result.get('error')}")

        except Exception as e:
            failed += 1
            print(f"   ❌ Error: {str(e)}")

    # Calculate aggregate metrics
    print("\n3️⃣  Calculating aggregate metrics...")

    if all_results:
        avg_bleu = sum(r['metrics']['bleu'] for r in all_results) / len(all_results)
        avg_rouge1 = sum(r['metrics']['rouge-1'] for r in all_results) / len(all_results)
        avg_rouge2 = sum(r['metrics']['rouge-2'] for r in all_results) / len(all_results)
        avg_rougeL = sum(r['metrics']['rouge-L'] for r in all_results) / len(all_results)
        avg_time = total_time / successful if successful > 0 else 0
        total_tokens = sum(r['tokens'] for r in all_results)

        summary = {
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
                'total_time': total_time,
                'total_tokens': total_tokens,
                'avg_tokens': total_tokens / successful if successful > 0 else 0
            },
            'detailed_results': all_results
        }

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n📊 Results:")
        print(f"   Total images: {len(test_images)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {successful/len(test_images)*100:.1f}%")

        print(f"\n📈 Quality Metrics:")
        print(f"   Average BLEU: {avg_bleu:.3f}")
        print(f"   Average ROUGE-1: {avg_rouge1:.3f}")
        print(f"   Average ROUGE-2: {avg_rouge2:.3f}")
        print(f"   Average ROUGE-L: {avg_rougeL:.3f}")

        print(f"\n⏱️  Performance:")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average tokens: {total_tokens / successful if successful > 0 else 0:.0f}")

        # Save results
        if save_results:
            output_file = "evaluation_results.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

        print("\n" + "=" * 70)

        return summary

    else:
        print("\n❌ No successful evaluations")
        return {}


def main():
    """Main evaluation"""

    # Load test data
    print("\n🔍 Loading test data...")

    # Use existing patient reports as references
    report_manager = ReportManager()
    all_reports = report_manager.get_all_reports()

    # Check for test images
    data_dir = Path("data/raw")

    # Find NORMAL images
    normal_dir = data_dir / "NORMAL"
    pneumonia_dir = data_dir / "PNEUMONIA"

    test_images = []
    test_reports = []

    # Use a few images for evaluation (not all 250 to save time)
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

    print(f"   Found {len(test_images)} test images")

    # Run evaluation
    results = evaluate_pipeline(test_images, test_reports)

    if results:
        print("\n✅ Evaluation complete!")
    else:
        print("\n❌ Evaluation failed")


if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

    main()
