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
        language: Optional[str] = None,
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        output_format: str = "wav",
    ) -> np.ndarray:
        """
        Generate speech synchronously.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            language: Language code for multilingual model (e.g., 'en', 'es', 'fr', 'ja', 'zh')
            temperature: Sampling temperature
            exaggeration: Emotion exaggeration
            output_format: Output audio format (wav, pcm_24000_16, mulaw_8000, etc.)

        Returns:
            numpy array of audio samples (float32)

        Example:
            >>> client = ChatterboxClient()
            >>> # English
            >>> audio = client.generate("Hello world!", language="en")
            >>> # Spanish
            >>> audio = client.generate("Hola mundo!", language="es")
            >>> # Japanese
            >>> audio = client.generate("こんにちは世界!", language="ja")
        """
        import requests

        data = {
            "text": text,
            "voice_id": voice_id,
            "temperature": temperature,
            "exaggeration": exaggeration,
            "output_format": output_format,
        }
        if language:
            data["language"] = language

        response = requests.post(
            f"{self.base_url}/v1/tts/generate",
            data=data,
        )
        response.raise_for_status()

        # Parse based on format
        if output_format == "wav":
            wav_bytes = response.content
            # Skip WAV header (44 bytes) and convert int16 to float32
            audio_int16 = np.frombuffer(wav_bytes[44:], dtype=np.int16)
            return audio_int16.astype(np.float32) / 32767.0
        elif "f32" in output_format:
            return np.frombuffer(response.content, dtype=np.float32)
        elif "mulaw" in output_format or "alaw" in output_format:
            # G.711 encoded - return raw bytes, user must decode
            return np.frombuffer(response.content, dtype=np.uint8)
        else:
            # Assume int16 PCM
            audio_int16 = np.frombuffer(response.content, dtype=np.int16)
            return audio_int16.astype(np.float32) / 32767.0

    def list_languages(self) -> list:
        """List supported languages (for multilingual model)."""
        import requests

        response = requests.get(f"{self.base_url}/v1/languages")
        response.raise_for_status()
        return response.json().get("languages", [])

    def list_formats(self) -> list:
        """List supported audio output formats."""
        import requests

        response = requests.get(f"{self.base_url}/v1/formats")
        response.raise_for_status()
        return response.json()["formats"]

    def stream_sync(
        self,
        text: str,
        voice_id: str = "default",
        chunk_size: int = 50,
        temperature: float = 0.8,
        output_format: str = "pcm_24000_16",
        language: Optional[str] = None,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Stream speech generation synchronously using SSE.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            chunk_size: Tokens per chunk
            temperature: Sampling temperature
            output_format: Output audio format (pcm_24000_16, mulaw_8000, etc.)
            language: Language code for multilingual model

        Yields:
            Tuples of (audio_chunk, metrics)
        """
        import requests

        params = {
            "text": text,
            "voice_id": voice_id,
            "chunk_size": chunk_size,
            "temperature": temperature,
            "output_format": output_format,
        }
        if language:
            params["language"] = language

        response = requests.get(
            f"{self.base_url}/v1/tts/stream-sse",
            params=params,
            stream=True,
        )
        response.raise_for_status()

        current_audio = None
        format_info = None
        event_type = None

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if line.startswith("event:"):
                event_type = line[7:].strip()
            elif line.startswith("data:"):
                data_str = line[6:].strip()

                if event_type == "format":
                    format_info = json.loads(data_str)

                elif event_type == "audio":
                    # New format with index
                    try:
                        audio_data = json.loads(data_str)
                        audio_bytes = base64.b64decode(audio_data["chunk"])
                    except (json.JSONDecodeError, KeyError):
                        # Fallback for old format
                        audio_bytes = base64.b64decode(data_str)

                    # Parse based on format
                    if format_info and format_info.get("encoding") == "mulaw":
                        current_audio = np.frombuffer(audio_bytes, dtype=np.uint8)
                    elif format_info and format_info.get("encoding") == "alaw":
                        current_audio = np.frombuffer(audio_bytes, dtype=np.uint8)
                    elif format_info and format_info.get("bits_per_sample") == 32:
                        current_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    else:
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        current_audio = audio_int16.astype(np.float32) / 32767.0

                elif event_type == "metrics" and current_audio is not None:
                    metrics = json.loads(data_str)
                    yield current_audio, metrics
                    current_audio = None

                elif event_type == "done":
                    break

                elif event_type == "error":
                    error_data = json.loads(data_str)
                    raise RuntimeError(error_data.get("error", "Unknown error"))

    async def stream(
        self,
        text: str,
        voice_id: str = "default",
        chunk_size: int = 50,
        temperature: float = 0.8,
        output_format: str = "pcm_24000_16",
    ) -> AsyncGenerator[Tuple[np.ndarray, dict], None]:
        """
        Stream speech generation asynchronously via WebSocket.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            chunk_size: Tokens per chunk
            temperature: Sampling temperature
            output_format: Output audio format (pcm_24000_16, mulaw_8000, etc.)

        Yields:
            Tuples of (audio_chunk, metrics)
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required for async streaming. Install with: pip install websockets"
            )

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")

        async with websockets.connect(f"{ws_url}/v1/tts/stream") as ws:
            # Send generate request
            await ws.send(
                json.dumps(
                    {
                        "action": "generate",
                        "text": text,
                        "voice_id": voice_id,
                        "chunk_size": chunk_size,
                        "temperature": temperature,
                        "output_format": output_format,
                    }
                )
            )

            current_audio = None
            format_info = None

            async for message in ws:
                if isinstance(message, bytes):
                    # Binary audio data - parse based on format
                    if format_info and format_info.get("encoding") == "mulaw":
                        current_audio = np.frombuffer(message, dtype=np.uint8)
                    elif format_info and format_info.get("encoding") == "alaw":
                        current_audio = np.frombuffer(message, dtype=np.uint8)
                    elif format_info and format_info.get("bits_per_sample") == 32:
                        current_audio = np.frombuffer(message, dtype=np.float32)
                    else:
                        audio_int16 = np.frombuffer(message, dtype=np.int16)
                        current_audio = audio_int16.astype(np.float32) / 32767.0
                else:
                    # JSON message
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "format":
                        format_info = data.get("data", {})

                    elif msg_type == "metrics" and current_audio is not None:
                        yield current_audio, data
                        current_audio = None

                    elif msg_type == "done":
                        break

                    elif msg_type == "stopped":
                        logger.info("Generation stopped by server")
                        break

                    elif msg_type == "error":
                        raise RuntimeError(data.get("message", "Unknown error"))

    async def stop_generation(self, ws) -> bool:
        """
        Stop the current generation on an active WebSocket.

        Args:
            ws: Active WebSocket connection

        Returns:
            True if stop was acknowledged
        """
        await ws.send(json.dumps({"action": "stop"}))
        response = await ws.recv()
        data = json.loads(response)
        return data.get("type") == "stopped"

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
        language: str = "en",
    ) -> dict:
        """
        Create a new voice from reference audio.

        Args:
            voice_id: Unique voice identifier
            audio_path: Path to reference audio file
            name: Optional display name
            language: Language code (default: "en")

        Returns:
            Voice creation response
        """
        import requests

        with open(audio_path, "rb") as f:
            files = {"audio_file": f}
            data = {"voice_id": voice_id, "language": language}
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

    def set_voice(self, voice_id: str) -> dict:
        """
        Set the current active voice.

        Args:
            voice_id: Voice ID to set as active

        Returns:
            Response confirming voice was set
        """
        import requests

        response = requests.post(f"{self.base_url}/v1/voices/{voice_id}/set")
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
