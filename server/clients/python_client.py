"""Python client SDK for Chatterbox TTS API."""

import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Generator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChatterboxClient:
    """
    Python client for the Chatterbox TTS API.

    Supports both synchronous and asynchronous operations.

    Example:
        >>> client = ChatterboxClient("http://localhost:8000")
        >>> audio = client.generate("Hello world!")
        >>> # Or async streaming:
        >>> async for chunk in client.stream("Hello world!"):
        ...     play_audio(chunk)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the Chatterbox API server
        """
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        text: str,
        voice_id: str = "default",
        temperature: float = 0.8,
        exaggeration: float = 0.0,
    ) -> np.ndarray:
        """
        Generate speech synchronously.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            temperature: Sampling temperature
            exaggeration: Emotion exaggeration

        Returns:
            numpy array of audio samples (float32, 24kHz)
        """
        import requests

        response = requests.post(
            f"{self.base_url}/v1/tts/generate",
            data={
                "text": text,
                "voice_id": voice_id,
                "temperature": temperature,
                "exaggeration": exaggeration,
            },
        )
        response.raise_for_status()

        # Parse WAV and extract audio data
        wav_bytes = response.content
        # Skip WAV header (44 bytes) and convert int16 to float32
        audio_int16 = np.frombuffer(wav_bytes[44:], dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        return audio_float

    def stream_sync(
        self,
        text: str,
        voice_id: str = "default",
        chunk_size: int = 50,
        temperature: float = 0.8,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Stream speech generation synchronously using SSE.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            chunk_size: Tokens per chunk
            temperature: Sampling temperature

        Yields:
            Tuples of (audio_chunk, metrics)
        """
        import requests

        response = requests.get(
            f"{self.base_url}/v1/tts/stream-sse",
            params={
                "text": text,
                "voice_id": voice_id,
                "chunk_size": chunk_size,
                "temperature": temperature,
            },
            stream=True,
        )
        response.raise_for_status()

        current_audio = None
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if line.startswith("event:"):
                event_type = line[7:].strip()
            elif line.startswith("data:"):
                data = line[6:].strip()

                if event_type == "audio":
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(data)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    current_audio = audio_int16.astype(np.float32) / 32767.0

                elif event_type == "metrics" and current_audio is not None:
                    metrics = json.loads(data)
                    yield current_audio, metrics
                    current_audio = None

                elif event_type == "done":
                    break

    async def stream(
        self,
        text: str,
        voice_id: str = "default",
        chunk_size: int = 50,
        temperature: float = 0.8,
    ) -> AsyncGenerator[Tuple[np.ndarray, dict], None]:
        """
        Stream speech generation asynchronously via WebSocket.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            chunk_size: Tokens per chunk
            temperature: Sampling temperature

        Yields:
            Tuples of (audio_chunk, metrics)
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required for async streaming. Install with: pip install websockets")

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")

        async with websockets.connect(f"{ws_url}/v1/tts/stream") as ws:
            # Send generate request
            await ws.send(json.dumps({
                "action": "generate",
                "text": text,
                "voice_id": voice_id,
                "chunk_size": chunk_size,
                "temperature": temperature,
            }))

            current_audio = None
            async for message in ws:
                if isinstance(message, bytes):
                    # Binary audio data
                    audio_int16 = np.frombuffer(message, dtype=np.int16)
                    current_audio = audio_int16.astype(np.float32) / 32767.0
                else:
                    # JSON message
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "metrics" and current_audio is not None:
                        yield current_audio, data
                        current_audio = None

                    elif msg_type == "done":
                        break

                    elif msg_type == "error":
                        raise RuntimeError(data.get("message", "Unknown error"))

    def list_voices(self) -> list:
        """List available voices."""
        import requests
        response = requests.get(f"{self.base_url}/v1/voices")
        response.raise_for_status()
        return response.json()["voices"]

    def create_voice(
        self,
        voice_id: str,
        audio_path: str,
        name: Optional[str] = None,
    ) -> dict:
        """
        Create a new voice from reference audio.

        Args:
            voice_id: Unique voice identifier
            audio_path: Path to reference audio file
            name: Optional display name

        Returns:
            Voice creation response
        """
        import requests

        with open(audio_path, "rb") as f:
            files = {"audio_file": f}
            data = {"voice_id": voice_id}
            if name:
                data["name"] = name

            response = requests.post(
                f"{self.base_url}/v1/voices/create",
                data=data,
                files=files,
            )
        response.raise_for_status()
        return response.json()

    def delete_voice(self, voice_id: str) -> dict:
        """Delete a voice."""
        import requests
        response = requests.delete(f"{self.base_url}/v1/voices/{voice_id}")
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict:
        """Check API health."""
        try:
            import aiohttp
        except ImportError:
            import requests
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()
