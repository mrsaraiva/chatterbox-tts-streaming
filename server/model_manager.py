"""Model manager for loading and caching TTS models."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
from threading import Lock

import torch

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Configuration for a voice."""
    name: str
    audio_path: str
    created_at: float = 0.0


class ModelManager:
    """
    Singleton manager for TTS models and voice cache.

    Handles:
    - Model loading on startup
    - Voice caching (pre-computed conditionals)
    - Request serialization for GPU access
    """

    _instance: Optional["ModelManager"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.model_type: str = "turbo"  # "turbo" or "standard"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.voices: Dict[str, Any] = {}  # voice_id -> Conditionals
        self.voice_configs: Dict[str, VoiceConfig] = {}
        self.voices_dir: Path = Path("./voices")
        self.request_lock = asyncio.Lock()
        self._initialized = True

    async def initialize(self, model_type: str = "turbo", device: Optional[str] = None):
        """
        Initialize the model manager and load the TTS model.

        Args:
            model_type: "turbo" for ChatterboxTurboTTS, "standard" for ChatterboxTTS
            device: Device to load model on (cuda/cpu/mps)
        """
        if device:
            self.device = device
        self.model_type = model_type

        logger.info(f"Loading {model_type} model on {self.device}...")

        # Load model in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        # Load cached voices
        await self._load_cached_voices()

        logger.info(f"Model loaded successfully. {len(self.voices)} voices available.")

    def _load_model(self):
        """Load the TTS model (runs in thread pool)."""
        if self.model_type == "turbo":
            from chatterbox import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(self.device)
        else:
            from chatterbox import ChatterboxTTS
            self.model = ChatterboxTTS.from_pretrained(self.device)

    async def _load_cached_voices(self):
        """Load pre-computed voice conditionals from disk."""
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # Load default voice if model has one
        if self.model.conds is not None:
            self.voices["default"] = self.model.conds
            self.voice_configs["default"] = VoiceConfig(
                name="Default Voice",
                audio_path="builtin",
            )

        # Load saved voices
        for voice_file in self.voices_dir.glob("*.pt"):
            voice_id = voice_file.stem
            try:
                from chatterbox.tts import Conditionals
                conds = Conditionals.load(voice_file, map_location=self.device)
                self.voices[voice_id] = conds
                self.voice_configs[voice_id] = VoiceConfig(
                    name=voice_id,
                    audio_path=str(voice_file),
                    created_at=voice_file.stat().st_mtime,
                )
                logger.info(f"Loaded voice: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to load voice {voice_id}: {e}")

    async def create_voice(
        self,
        voice_id: str,
        audio_path: str,
        name: Optional[str] = None,
    ) -> VoiceConfig:
        """
        Create a new voice from reference audio.

        Args:
            voice_id: Unique identifier for the voice
            audio_path: Path to reference audio file (6-15 seconds)
            name: Optional display name

        Returns:
            VoiceConfig for the created voice
        """
        async with self.request_lock:
            # Prepare conditionals from audio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.model.prepare_conditionals,
                audio_path,
            )

            # Cache conditionals
            conds = self.model.conds
            self.voices[voice_id] = conds

            # Save to disk
            voice_file = self.voices_dir / f"{voice_id}.pt"
            conds.save(voice_file)

            config = VoiceConfig(
                name=name or voice_id,
                audio_path=audio_path,
                created_at=voice_file.stat().st_mtime,
            )
            self.voice_configs[voice_id] = config

            logger.info(f"Created voice: {voice_id}")
            return config

    async def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice.

        Args:
            voice_id: Voice identifier to delete

        Returns:
            True if deleted, False if not found
        """
        if voice_id == "default":
            raise ValueError("Cannot delete default voice")

        if voice_id not in self.voices:
            return False

        # Remove from cache
        del self.voices[voice_id]
        del self.voice_configs[voice_id]

        # Remove from disk
        voice_file = self.voices_dir / f"{voice_id}.pt"
        if voice_file.exists():
            voice_file.unlink()

        logger.info(f"Deleted voice: {voice_id}")
        return True

    def list_voices(self) -> Dict[str, VoiceConfig]:
        """List all available voices."""
        return self.voice_configs.copy()

    def get_voice(self, voice_id: str) -> Optional[Any]:
        """Get cached conditionals for a voice."""
        return self.voices.get(voice_id)

    async def set_voice(self, voice_id: str):
        """Set the active voice on the model."""
        if voice_id not in self.voices:
            raise ValueError(f"Voice not found: {voice_id}")
        self.model.conds = self.voices[voice_id]


# Global instance
model_manager = ModelManager()
