"""
Audio Processor Module for MedAssist Copilot
Handles voice input, transcription using Whisper, and audio processing
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import tempfile

import numpy as np

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor for voice input and transcription
    Uses OpenAI's Whisper model for speech-to-text
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None
    ):
        """
        Initialize the audio processor

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda' or 'cpu')
            sample_rate: Audio sample rate in Hz
        """
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.sample_rate = sample_rate or config.AUDIO_SAMPLE_RATE

        # Determine device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing Audio Processor...")
        logger.info(f"  Whisper model: {self.model_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Sample rate: {self.sample_rate} Hz")

        # Load Whisper model (lazy loading)
        self.model = None
        self.model_loaded = False

        # Statistics
        self.stats = {
            'total_transcriptions': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'average_rtf': 0.0  # Real-time factor
        }

    def load_model(self) -> bool:
        """
        Load the Whisper model

        Returns:
            True if successful, False otherwise
        """
        if self.model_loaded:
            logger.info("Model already loaded")
            return True

        try:
            import whisper

            logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()

            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )

            load_time = time.time() - start_time
            self.model_loaded = True

            logger.info(f"✅ Whisper model loaded in {load_time:.2f}s")
            return True

        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            return False

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        task: str = "transcribe"
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: 'transcribe' or 'translate'

        Returns:
            Dictionary with transcription and metadata, or None if failed
        """
        # Load model if not already loaded
        if not self.model_loaded:
            if not self.load_model():
                return None

        try:
            logger.info(f"Transcribing: {Path(audio_path).name}")
            start_time = time.time()

            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=False
            )

            process_time = time.time() - start_time

            # Get audio duration
            import librosa
            try:
                audio_duration = librosa.get_duration(filename=audio_path)
            except:
                # Fallback if librosa fails
                audio_duration = process_time

            # Calculate real-time factor
            rtf = process_time / audio_duration if audio_duration > 0 else 0

            # Update statistics
            self.stats['total_transcriptions'] += 1
            self.stats['total_audio_duration'] += audio_duration
            self.stats['total_processing_time'] += process_time
            self.stats['average_rtf'] = (
                self.stats['total_processing_time'] /
                self.stats['total_audio_duration']
                if self.stats['total_audio_duration'] > 0 else 0
            )

            logger.info(f"✅ Transcribed in {process_time:.2f}s (RTF: {rtf:.2f}x)")

            return {
                'text': result['text'].strip(),
                'language': result.get('language', language),
                'segments': result.get('segments', []),
                'audio_duration': audio_duration,
                'processing_time': process_time,
                'rtf': rtf
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Transcribe audio with word-level timestamps

        Args:
            audio_path: Path to audio file
            language: Language code

        Returns:
            List of segments with timestamps, or None if failed
        """
        result = self.transcribe_audio(audio_path, language=language)

        if not result:
            return None

        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('no_speech_prob', 0.0)
            })

        return segments

    def record_audio(
        self,
        duration: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds (None for continuous)
            output_path: Path to save audio file

        Returns:
            Path to recorded audio file, or None if failed
        """
        try:
            import sounddevice as sd
            import soundfile as sf

            # Create output path if not provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(
                    temp_dir,
                    f"recording_{int(time.time())}.wav"
                )

            logger.info(f"Recording audio...")
            if duration:
                logger.info(f"  Duration: {duration}s")

            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate) if duration else None,
                samplerate=self.sample_rate,
                channels=config.AUDIO_CHANNELS,
                dtype='float32'
            )

            if duration:
                sd.wait()  # Wait until recording is finished
            else:
                logger.info("  Press Ctrl+C to stop recording")
                try:
                    sd.wait()
                except KeyboardInterrupt:
                    logger.info("  Recording stopped")

            # Save to file
            sf.write(output_path, recording, self.sample_rate)

            logger.info(f"✅ Audio saved to: {output_path}")
            return output_path

        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice soundfile")
            return None
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return None

    def process_voice_command(
        self,
        audio_path: str,
        commands: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process voice command and extract intent

        Args:
            audio_path: Path to audio file
            commands: Dictionary of command keywords and their actions

        Returns:
            Dictionary with transcription and detected command
        """
        if commands is None:
            commands = config.VOICE_COMMANDS

        # Transcribe audio
        result = self.transcribe_audio(audio_path)

        if not result:
            return {
                'success': False,
                'error': 'Transcription failed'
            }

        text = result['text'].lower()

        # Detect command
        detected_command = None
        for keyword, action in commands.items():
            if keyword in text:
                detected_command = action
                break

        return {
            'success': True,
            'text': result['text'],
            'transcription': result,
            'command': detected_command,
            'text_lower': text
        }

    def create_voice_note(
        self,
        audio_path: str,
        section: str = "findings"
    ) -> Optional[Dict[str, Any]]:
        """
        Create a voice note for report editing

        Args:
            audio_path: Path to audio file
            section: Report section to append to

        Returns:
            Dictionary with note and metadata
        """
        result = self.transcribe_audio(audio_path)

        if not result:
            return None

        return {
            'section': section,
            'text': result['text'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'audio_duration': result['audio_duration'],
            'language': result['language']
        }

    def batch_transcribe(
        self,
        audio_files: List[str],
        language: Optional[str] = "en"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files

        Args:
            audio_files: List of audio file paths
            language: Language code

        Returns:
            List of transcription results
        """
        results = []

        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"Transcribing {i}/{len(audio_files)}")

            result = self.transcribe_audio(audio_path, language=language)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
            self.model_loaded = False

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded from memory")


# Utility functions

def quick_transcribe(audio_path: str, model_size: str = "base") -> Optional[str]:
    """
    Quick function to transcribe an audio file

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size

    Returns:
        Transcribed text or None
    """
    processor = AudioProcessor(model_size=model_size)
    result = processor.transcribe_audio(audio_path)
    return result['text'] if result else None


# Main execution for testing
if __name__ == "__main__":
    print("=" * 70)
    print("MedAssist Copilot - Audio Processor Test")
    print("=" * 70)

    # Initialize processor
    print("\n1️⃣  Initializing Audio Processor...")
    processor = AudioProcessor(model_size="base")
    print("   ✅ Audio processor initialized")

    # Check if test audio file exists
    print("\n2️⃣  Checking for test audio files...")
    test_audio_dir = Path("data/audio")

    if test_audio_dir.exists():
        audio_files = list(test_audio_dir.glob("*.wav")) + list(test_audio_dir.glob("*.mp3"))

        if audio_files:
            print(f"   Found {len(audio_files)} audio file(s)")

            # Load model
            print("\n3️⃣  Loading Whisper model...")
            if not processor.load_model():
                print("   ❌ Failed to load model")
                exit(1)

            # Test transcription
            print("\n4️⃣  Testing transcription...")
            test_file = audio_files[0]
            print(f"   File: {test_file.name}")

            result = processor.transcribe_audio(str(test_file))

            if result:
                print("\n   ✅ Transcription successful!")
                print(f"   Text: {result['text']}")
                print(f"   Duration: {result['audio_duration']:.2f}s")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   RTF: {result['rtf']:.2f}x")
            else:
                print("   ❌ Transcription failed")

        else:
            print("   ℹ️  No audio files found in data/audio/")
    else:
        print("   ℹ️  data/audio/ directory does not exist")

    print("\n💡 Usage Examples:")
    print("   1. Record audio:")
    print("      processor.record_audio(duration=5, output_path='test.wav')")
    print("\n   2. Transcribe audio:")
    print("      result = processor.transcribe_audio('test.wav')")
    print("\n   3. Create voice note:")
    print("      note = processor.create_voice_note('test.wav', section='findings')")

    # Show statistics
    print("\n5️⃣  Audio Processor Statistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("Audio processor test complete!")
    print("=" * 70)
    print("\n📝 Note: To test voice recording, you'll need a microphone")
    print("   and run: processor.record_audio(duration=5)")
