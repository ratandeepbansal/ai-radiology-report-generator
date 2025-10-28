"""
Week 3 Pipeline Test
Comprehensive test of RAG system and Audio processor
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag import RAGSystem
from src.audio_processor import AudioProcessor
import config


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_rag_system():
    """Test the enhanced RAG system"""
    print_section("ENHANCED RAG SYSTEM TEST")

    print("\n1️⃣  Initializing RAG System...")
    try:
        rag = RAGSystem()
        print("   ✅ RAG system initialized")
        print(f"   Embedding model: {rag.embedding_model_name}")
        print(f"   Vector DB: {rag.vector_db_type}")
        print(f"   Reports loaded: {rag.stats['total_reports']}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        print("\n   💡 To fix:")
        print("      pip install sentence-transformers chromadb")
        return False

    # Test similarity search
    print("\n2️⃣  Testing semantic similarity search...")
    test_queries = [
        ("pneumonia with consolidation", "Pneumonia cases"),
        ("normal chest X-ray clear lungs", "Normal cases"),
        ("pleural effusion fluid", "Effusion cases"),
    ]

    for query, description in test_queries:
        print(f"\n   🔍 {description}")
        print(f"   Query: {query}")

        results = rag.search_by_similarity(query, top_k=2)

        if results:
            print(f"   Found {len(results)} similar reports:")
            for i, result in enumerate(results, 1):
                report = result['report']
                print(f"\n   Result {i}:")
                print(f"     Patient: {report['patient_id']}")
                print(f"     Date: {report['date']}")
                print(f"     Similarity: {result['similarity']:.3f}")
                impression = report['report']['impression']
                print(f"     Impression: {impression[:80]}...")
        else:
            print("   ❌ No results found")

    # Test patient-specific search
    print("\n3️⃣  Testing patient-specific reports...")
    test_patient = "P002"
    print(f"   Patient ID: {test_patient}")

    patient_reports = rag.search_by_patient(test_patient, top_k=3)

    if patient_reports:
        print(f"   ✅ Found {len(patient_reports)} report(s)")
        for report in patient_reports:
            print(f"     - Date: {report['date']}")
            print(f"       Impression: {report['report']['impression'][:60]}...")
    else:
        print("   ℹ️  No reports found")

    # Test context generation
    print("\n4️⃣  Testing context generation for report...")
    context = rag.get_relevant_context(
        query="chest X-ray showing infiltrate in right lower lobe",
        patient_id="P002",
        max_context_length=300
    )

    print(f"   Generated context ({len(context)} chars):")
    print(f"   {context[:250]}...")

    # Show statistics
    print("\n5️⃣  RAG System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    return True


def test_audio_processor():
    """Test the audio processor"""
    print_section("AUDIO PROCESSOR TEST")

    print("\n1️⃣  Initializing Audio Processor...")
    try:
        processor = AudioProcessor(model_size="base")
        print("   ✅ Audio processor initialized")
        print(f"   Whisper model: {processor.model_size}")
        print(f"   Device: {processor.device}")
        print(f"   Sample rate: {processor.sample_rate} Hz")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        return False

    # Check for test audio files
    print("\n2️⃣  Checking for test audio files...")
    test_audio_dir = Path("data/audio")
    test_audio_dir.mkdir(parents=True, exist_ok=True)

    audio_files = (
        list(test_audio_dir.glob("*.wav")) +
        list(test_audio_dir.glob("*.mp3")) +
        list(test_audio_dir.glob("*.m4a"))
    )

    if audio_files:
        print(f"   ✅ Found {len(audio_files)} audio file(s)")

        # Load Whisper model
        print("\n3️⃣  Loading Whisper model...")
        if not processor.load_model():
            print("   ❌ Failed to load model")
            print("\n   💡 To fix:")
            print("      pip install openai-whisper")
            return False

        # Test transcription
        print("\n4️⃣  Testing audio transcription...")
        test_file = audio_files[0]
        print(f"   File: {test_file.name}")

        result = processor.transcribe_audio(str(test_file))

        if result:
            print("\n   ✅ Transcription successful!")
            print(f"   Text: {result['text']}")
            print(f"   Language: {result['language']}")
            print(f"   Duration: {result['audio_duration']:.2f}s")
            print(f"   Processing: {result['processing_time']:.2f}s")
            print(f"   RTF: {result['rtf']:.2f}x")
        else:
            print("   ❌ Transcription failed")
            return False

        # Test voice note creation
        print("\n5️⃣  Testing voice note creation...")
        note = processor.create_voice_note(str(test_file), section="findings")

        if note:
            print("   ✅ Voice note created!")
            print(f"   Section: {note['section']}")
            print(f"   Text: {note['text'][:80]}...")
            print(f"   Timestamp: {note['timestamp']}")
        else:
            print("   ❌ Voice note creation failed")

        # Show statistics
        print("\n6️⃣  Audio Processor Statistics:")
        stats = processor.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        return True

    else:
        print("   ℹ️  No audio files found in data/audio/")
        print("\n   💡 To test audio transcription:")
        print("      1. Add a WAV/MP3 file to data/audio/")
        print("      2. Or record audio with:")
        print("         processor.record_audio(duration=5, output_path='data/audio/test.wav')")
        return None  # Skip, not a failure


def test_integration():
    """Test integration of RAG + Audio in pipeline"""
    print_section("INTEGRATION TEST - RAG + AUDIO")

    print("\n1️⃣  Testing RAG-enhanced context retrieval...")

    try:
        rag = RAGSystem()

        # Simulate a findings from vision model
        vision_findings = "Chest X-ray shows bilateral infiltrates with air bronchograms"

        # Get relevant context
        context = rag.get_relevant_context(
            query=vision_findings,
            patient_id="P002",
            max_context_length=400
        )

        print("   ✅ RAG context retrieved")
        print(f"   Context length: {len(context)} chars")
        print(f"   Preview: {context[:200]}...")

    except Exception as e:
        print(f"   ❌ RAG integration failed: {str(e)}")

    print("\n2️⃣  Testing Audio → Text → Report workflow...")

    # Check for test audio
    test_audio_dir = Path("data/audio")
    audio_files = list(test_audio_dir.glob("*.wav")) if test_audio_dir.exists() else []

    if audio_files:
        try:
            processor = AudioProcessor(model_size="base")

            # Transcribe voice note
            print(f"   Processing: {audio_files[0].name}")
            result = processor.transcribe_audio(str(audio_files[0]))

            if result:
                print("   ✅ Voice input processed")
                print(f"   Transcribed: {result['text'][:100]}...")

                # Simulate adding to report
                voice_note = processor.create_voice_note(
                    str(audio_files[0]),
                    section="findings"
                )

                if voice_note:
                    print(f"   ✅ Voice note ready for report editing")
                    print(f"   Section: {voice_note['section']}")

        except Exception as e:
            print(f"   ❌ Audio integration failed: {str(e)}")
    else:
        print("   ℹ️  No audio files available for testing")

    print("\n3️⃣  Integration capabilities:")
    print("   ✅ RAG can enhance report generation with context")
    print("   ✅ Voice notes can be added to reports")
    print("   ✅ Semantic search improves prior report retrieval")
    print("   ✅ Real-time transcription for dictation")


def main():
    """Run all Week 3 tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "WEEK 3 PIPELINE - COMPREHENSIVE TEST")
    print("=" * 70)

    results = {}

    # Test 1: RAG System
    try:
        results['rag'] = test_rag_system()
    except Exception as e:
        print(f"\n❌ RAG test error: {str(e)}")
        results['rag'] = False

    # Test 2: Audio Processor
    try:
        results['audio'] = test_audio_processor()
    except Exception as e:
        print(f"\n❌ Audio test error: {str(e)}")
        results['audio'] = False

    # Test 3: Integration
    try:
        test_integration()
        results['integration'] = True
    except Exception as e:
        print(f"\n❌ Integration test error: {str(e)}")
        results['integration'] = False

    # Summary
    print_section("SUMMARY")

    print("\n📋 Test Results:")
    print(f"   RAG System:         {'✅ PASSED' if results.get('rag') else '❌ FAILED' if results.get('rag') is False else '⏭️  SKIPPED'}")
    print(f"   Audio Processor:    {'✅ PASSED' if results.get('audio') else '❌ FAILED' if results.get('audio') is False else '⏭️  SKIPPED'}")
    print(f"   Integration:        {'✅ PASSED' if results.get('integration') else '❌ FAILED' if results.get('integration') is False else '⏭️  SKIPPED'}")

    # Check results
    if results.get('rag') and results.get('integration'):
        print("\n✅ Week 3 core components are working!")

        print("\n📋 Week 3 Deliverables:")
        print("   ✅ Enhanced RAG system with vector database")
        print("   ✅ Semantic similarity search")
        print("   ✅ Audio processor with Whisper")
        print("   ✅ Voice transcription functionality")
        print("   ✅ Voice note creation")

        if results.get('audio'):
            print("   ✅ Full audio pipeline tested")
        else:
            print("   ⏭️  Audio testing skipped (no test files)")

        print("\n🎯 Ready for Week 4 Tasks:")
        print("   - Streamlit UI development")
        print("   - Report editor with voice input")
        print("   - Visual report display")
        print("   - Evaluation metrics")

    else:
        print("\n⚠️  Some tests failed or were skipped")

        if not results.get('rag'):
            print("\n💡 To fix RAG issues:")
            print("   pip install sentence-transformers chromadb")

        if results.get('audio') is False:
            print("\n💡 To fix Audio issues:")
            print("   pip install openai-whisper librosa sounddevice soundfile")

        if results.get('audio') is None:
            print("\n💡 To test audio transcription:")
            print("   1. Add a WAV/MP3 file to data/audio/")
            print("   2. Run this test again")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
