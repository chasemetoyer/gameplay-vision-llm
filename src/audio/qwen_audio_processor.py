"""
Qwen2-Audio Processor Module.

This module implements dual-mode audio analysis using Qwen2-Audio-Instruct:
1. Automatic Speech Recognition (ASR) for speech transcription
2. Non-speech audio analysis for event detection (music, effects, ambient)

References:
- [B: 69, B: 70] Qwen2-Audio dual-mode operation
- [A: 32] Timestamp marking for timeline integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AudioEventType(Enum):
    """Types of detected audio events."""

    SPEECH = "speech"  # Spoken dialog
    MUSIC = "music"  # Background music
    EFFECT = "effect"  # Sound effects (explosions, etc.)
    AMBIENT = "ambient"  # Environmental sounds
    UI = "ui"  # UI sounds (clicks, notifications)
    SILENCE = "silence"  # No significant audio


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech."""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker_id: Optional[str] = None  # For multi-speaker scenarios

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_timeline_entry(self) -> str:
        """Format for timeline integration."""
        speaker = f"[{self.speaker_id}]" if self.speaker_id else "[Speech]"
        return f'{speaker}: "{self.text}"'


@dataclass
class AudioEvent:
    """A detected non-speech audio event."""

    event_type: AudioEventType
    description: str  # Natural language description
    start_time: float
    end_time: float
    confidence: float = 1.0
    intensity: float = 0.5  # Relative loudness 0-1
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_timeline_entry(self) -> str:
        """Format for timeline integration."""
        return f"(Audio: {self.description})"


@dataclass
class AudioAnalysisResult:
    """Complete analysis result for an audio segment."""

    start_time: float
    end_time: float
    transcriptions: list[TranscriptionSegment] = field(default_factory=list)
    events: list[AudioEvent] = field(default_factory=list)
    dominant_type: AudioEventType = AudioEventType.SILENCE

    @property
    def has_speech(self) -> bool:
        return len(self.transcriptions) > 0

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0

    def get_timeline_entries(self) -> list[tuple[float, str]]:
        """Get all entries formatted for timeline."""
        entries = []

        for seg in self.transcriptions:
            entries.append((seg.start_time, seg.to_timeline_entry()))

        for event in self.events:
            entries.append((event.start_time, event.to_timeline_entry()))

        entries.sort(key=lambda x: x[0])
        return entries


@dataclass
class QwenAudioConfig:
    """Configuration for Qwen2-Audio processor."""

    # Model settings
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Audio settings
    sample_rate: int = 16000  # Expected sample rate
    max_audio_length: float = 30.0  # Max seconds per chunk

    # ASR settings
    language: str = "en"  # Primary language for transcription
    enable_timestamps: bool = True  # Word-level timestamps
    min_speech_duration: float = 0.5  # Minimum speech segment length

    # Event detection settings
    event_detection_prompt: str = "Describe the non-speech sounds in this audio."
    min_event_confidence: float = 0.3

    # Performance
    use_flash_attention: bool = True
    batch_size: int = 4


class AudioPreprocessor:
    """Preprocesses audio for Qwen2-Audio input."""

    def __init__(self, config: QwenAudioConfig):
        self.config = config

    def load_audio(
        self,
        audio_path: str,
    ) -> tuple[NDArray[np.float32], int]:
        """
        Load audio file and resample if needed.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            return audio.astype(np.float32), sr
        except ImportError:
            try:
                import soundfile as sf

                audio, sr = sf.read(audio_path)
                if sr != self.config.sample_rate:
                    # Simple resampling
                    import scipy.signal

                    ratio = self.config.sample_rate / sr
                    audio = scipy.signal.resample(
                        audio, int(len(audio) * ratio)
                    )
                return audio.astype(np.float32), self.config.sample_rate
            except ImportError:
                logger.warning("No audio library available")
                return np.zeros(self.config.sample_rate, dtype=np.float32), self.config.sample_rate

    def extract_from_video(
        self,
        video_path: str,
    ) -> tuple[NDArray[np.float32], int]:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import subprocess
            import tempfile

            # Use ffmpeg to extract audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.config.sample_rate),
                "-ac", "1",
                tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            audio, sr = self.load_audio(tmp_path)

            import os
            os.unlink(tmp_path)

            return audio, sr
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to extract audio from video: {e}")
            return np.zeros(self.config.sample_rate, dtype=np.float32), self.config.sample_rate

    def chunk_audio(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        chunk_duration: Optional[float] = None,
        overlap: float = 0.5,
    ) -> list[tuple[NDArray[np.float32], float, float]]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            chunk_duration: Duration per chunk (uses config default if None)
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of (chunk, start_time, end_time) tuples
        """
        chunk_duration = chunk_duration or self.config.max_audio_length
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        stride = chunk_samples - overlap_samples

        chunks = []
        total_samples = len(audio)

        pos = 0
        while pos < total_samples:
            end_pos = min(pos + chunk_samples, total_samples)
            chunk = audio[pos:end_pos]

            # Pad if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            start_time = pos / sample_rate
            end_time = end_pos / sample_rate

            chunks.append((chunk, start_time, end_time))
            pos += stride

        return chunks

    def compute_mel_spectrogram(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Compute mel spectrogram for audio analysis."""
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=128,
                fmax=8000,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            return mel_db.astype(np.float32)
        except ImportError:
            # Simple placeholder
            return np.zeros((128, len(audio) // 512), dtype=np.float32)


class QwenAudioModel:
    """
    Wrapper for Qwen2-Audio model using official HuggingFace API.
    
    Uses Qwen2AudioForConditionalGeneration with apply_chat_template()
    for proper conversation formatting per official documentation.
    """

    def __init__(self, config: QwenAudioConfig):
        self.config = config
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy load Qwen2-Audio model with official API."""
        if self._model is not None:
            return

        # Try official Qwen2AudioForConditionalGeneration first
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

            logger.info(f"Loading Qwen2-Audio: {self.config.model_name}")
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Use official Qwen2AudioForConditionalGeneration class
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                device_map="auto",
                attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
            )
            self._model.eval()
            
            # Update sample rate from processor
            if hasattr(self._processor, 'feature_extractor'):
                self.config.sample_rate = self._processor.feature_extractor.sampling_rate
                
            logger.info("Qwen2-Audio loaded successfully")
            return
        except ImportError:
            logger.info("Qwen2AudioForConditionalGeneration not found, trying AutoModel fallback...")
        except Exception as e:
            logger.info(f"Qwen2AudioForConditionalGeneration failed: {e}, trying fallback...")
        
        # Fallback: Use AutoModelForCausalLM with trust_remote_code
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading Qwen2-Audio with AutoModel fallback: {self.config.model_name}")
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            
            if hasattr(self._processor, 'feature_extractor'):
                self.config.sample_rate = self._processor.feature_extractor.sampling_rate
                
            logger.info("Qwen2-Audio loaded successfully (via AutoModel)")
            return
        except Exception as e:
            logger.info(f"AutoModel fallback failed: {e}, trying Whisper...")
        
        # Final fallback: Use OpenAI Whisper for transcription
        try:
            import whisper
            
            logger.info("Loading OpenAI Whisper as audio fallback...")
            self._model = whisper.load_model("base", device=self.config.device)
            self._processor = "whisper"  # Flag for whisper mode
            self._whisper_mode = True
            logger.info("OpenAI Whisper loaded successfully as fallback")
            return
        except ImportError:
            logger.warning("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.warning(f"Whisper failed: {e}")
        
        # All fallbacks failed
        logger.warning("All audio backends failed - audio analysis disabled")
        self._model = "placeholder"
        self._processor = None

    def transcribe(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> str:
        """
        Transcribe speech in audio.
        
        Uses Qwen2-Audio if available, falls back to Whisper.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        self._load_model()

        if self._model == "placeholder":
            return ""

        # Whisper mode
        if getattr(self, '_whisper_mode', False):
            try:
                import tempfile
                import soundfile as sf
                
                # Whisper expects file input, save temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, sample_rate)
                    result = self._model.transcribe(f.name, language="en")
                    import os
                    os.unlink(f.name)
                    return result.get("text", "").strip()
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                return ""

        # Qwen2-Audio mode
        try:
            # Build conversation in official format
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": "Transcribe the speech in this audio."},
                    ],
                },
            ]

            # Apply chat template (official API)
            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Process inputs
            inputs = self._processor(
                text=text,
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generate_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
                # Remove input tokens from output
                generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

            # Decode
            response = self._processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def analyze_audio_events(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze non-speech audio events using official chat template API.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            prompt: Custom analysis prompt
            
        Returns:
            Description of audio events
        """
        self._load_model()

        if self._model == "placeholder":
            return "No audio events detected."

        prompt = prompt or self.config.event_detection_prompt

        try:
            # Build conversation in official format
            conversation = [
                {"role": "system", "content": "You are a helpful assistant that analyzes audio."},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template (official API)
            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self._processor(
                text=text,
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generate_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
                generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

            response = self._processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return "Analysis failed."


class AudioEventParser:
    """Parses model descriptions into structured AudioEvent objects."""

    # Keywords for event classification
    EVENT_KEYWORDS = {
        AudioEventType.MUSIC: ["music", "melody", "song", "tune", "beat", "rhythm"],
        AudioEventType.EFFECT: ["explosion", "crash", "bang", "hit", "impact", "shot"],
        AudioEventType.AMBIENT: ["wind", "rain", "water", "birds", "traffic", "crowd"],
        AudioEventType.UI: ["click", "beep", "notification", "chime", "alert"],
    }

    def parse_description(
        self,
        description: str,
        start_time: float,
        end_time: float,
    ) -> list[AudioEvent]:
        """
        Parse a text description into AudioEvent objects.
        
        Args:
            description: Model-generated description
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            List of AudioEvent objects
        """
        events = []
        desc_lower = description.lower()

        # Check for each event type
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    events.append(
                        AudioEvent(
                            event_type=event_type,
                            description=description,
                            start_time=start_time,
                            end_time=end_time,
                            confidence=0.8,
                        )
                    )
                    break

        # Default to ambient if nothing specific found
        if not events and description and "silence" not in desc_lower:
            events.append(
                AudioEvent(
                    event_type=AudioEventType.AMBIENT,
                    description=description,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.5,
                )
            )

        return events


class QwenAudioProcessor:
    """
    Main interface for Qwen2-Audio dual-mode processing.
    
    Provides:
    1. ASR transcription for speech
    2. Audio event detection for non-speech
    3. Timeline-ready output format
    
    Example:
        >>> processor = QwenAudioProcessor()
        >>> 
        >>> # Analyze video audio
        >>> result = processor.analyze_video_audio("gameplay.mp4")
        >>> 
        >>> # Get transcriptions
        >>> for seg in result.transcriptions:
        ...     print(f"[{seg.start_time:.1f}s] {seg.text}")
        >>> 
        >>> # Get audio events
        >>> for event in result.events:
        ...     print(f"[{event.start_time:.1f}s] {event.description}")
    """

    def __init__(
        self,
        config: Optional[QwenAudioConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the Qwen Audio Processor.

        Args:
            config: Audio configuration. Uses defaults if not provided.
            device: Override device from config.
        """
        self.config = config or QwenAudioConfig()
        if device:
            self.config.device = device

        self.preprocessor = AudioPreprocessor(self.config)
        self.model = QwenAudioModel(self.config)
        self.event_parser = AudioEventParser()

        logger.info(f"QwenAudioProcessor initialized with device={self.config.device}")

    def transcribe(
        self,
        audio: Union[str, NDArray[np.float32]],
        sample_rate: Optional[int] = None,
    ) -> list[TranscriptionSegment]:
        """
        Transcribe speech from audio.
        
        Args:
            audio: Audio file path or waveform array
            sample_rate: Sample rate (required if audio is array)
            
        Returns:
            List of TranscriptionSegment objects
        """
        # Load audio if path
        if isinstance(audio, str):
            audio_array, sr = self.preprocessor.load_audio(audio)
        else:
            audio_array = audio
            sr = sample_rate or self.config.sample_rate

        # Chunk for long audio
        chunks = self.preprocessor.chunk_audio(audio_array, sr)

        segments = []
        for chunk, start_time, end_time in chunks:
            text = self.model.transcribe(chunk, sr)

            if text.strip():
                segments.append(
                    TranscriptionSegment(
                        text=text.strip(),
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.9,  # Placeholder
                    )
                )

        return segments

    def detect_audio_events(
        self,
        audio: Union[str, NDArray[np.float32]],
        sample_rate: Optional[int] = None,
    ) -> list[AudioEvent]:
        """
        Detect non-speech audio events.
        
        Args:
            audio: Audio file path or waveform array
            sample_rate: Sample rate (required if audio is array)
            
        Returns:
            List of AudioEvent objects
        """
        # Load audio if path
        if isinstance(audio, str):
            audio_array, sr = self.preprocessor.load_audio(audio)
        else:
            audio_array = audio
            sr = sample_rate or self.config.sample_rate

        # Chunk for long audio
        chunks = self.preprocessor.chunk_audio(audio_array, sr)

        events = []
        for chunk, start_time, end_time in chunks:
            description = self.model.analyze_audio_events(chunk, sr)
            chunk_events = self.event_parser.parse_description(
                description, start_time, end_time
            )
            events.extend(chunk_events)

        return events

    def analyze_audio(
        self,
        audio: Union[str, NDArray[np.float32]],
        sample_rate: Optional[int] = None,
    ) -> AudioAnalysisResult:
        """
        Perform full audio analysis (speech + events).
        
        Args:
            audio: Audio file path or waveform array
            sample_rate: Sample rate (required if audio is array)
            
        Returns:
            AudioAnalysisResult with all detections
        """
        # Load audio if path
        if isinstance(audio, str):
            audio_array, sr = self.preprocessor.load_audio(audio)
        else:
            audio_array = audio
            sr = sample_rate or self.config.sample_rate

        # Get total duration
        duration = len(audio_array) / sr

        # Run both analyses
        transcriptions = self.transcribe(audio_array, sr)
        events = self.detect_audio_events(audio_array, sr)

        # Determine dominant type
        if transcriptions:
            dominant = AudioEventType.SPEECH
        elif events:
            # Most common event type
            type_counts: dict[AudioEventType, int] = {}
            for e in events:
                type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            dominant = max(type_counts, key=type_counts.get)
        else:
            dominant = AudioEventType.SILENCE

        return AudioAnalysisResult(
            start_time=0.0,
            end_time=duration,
            transcriptions=transcriptions,
            events=events,
            dominant_type=dominant,
        )

    def analyze_video_audio(
        self,
        video_path: str,
    ) -> AudioAnalysisResult:
        """
        Extract and analyze audio from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            AudioAnalysisResult for the video's audio track
        """
        logger.info(f"Extracting audio from: {video_path}")
        audio, sr = self.preprocessor.extract_from_video(video_path)
        return self.analyze_audio(audio, sr)

    def get_timeline_events(
        self,
        result: AudioAnalysisResult,
    ) -> list[tuple[float, str]]:
        """
        Get timeline-ready events from analysis result.
        
        Args:
            result: Audio analysis result
            
        Returns:
            List of (timestamp, description) tuples
        """
        return result.get_timeline_entries()


def create_audio_processor(
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    device: str = "cuda",
) -> QwenAudioProcessor:
    """
    Factory function to create a Qwen Audio Processor.

    Args:
        model_name: HuggingFace model identifier
        device: Compute device

    Returns:
        Configured QwenAudioProcessor instance
    """
    config = QwenAudioConfig(model_name=model_name, device=device)
    return QwenAudioProcessor(config=config)
