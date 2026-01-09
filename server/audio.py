"""
Audio format utilities for Chatterbox TTS streaming server.

Provides support for multiple audio output formats including:
- WAV (complete file with header)
- PCM at various sample rates (24kHz, 16kHz, 8kHz)
- G.711 mu-law (Twilio, Vonage)
- G.711 A-law (European telephony)
"""

import io
import struct
from enum import Enum
from typing import Dict, Any, Optional

import numpy as np


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    WAV = "wav"  # Complete WAV file (44-byte header + PCM)
    PCM_24000_16 = "pcm_24000_16"  # Native 24kHz 16-bit signed PCM
    PCM_24000_F32 = "pcm_24000_f32"  # Native 24kHz 32-bit float PCM
    PCM_16000_16 = "pcm_16000_16"  # 16kHz 16-bit (speech recognition)
    PCM_8000_16 = "pcm_8000_16"  # 8kHz 16-bit (telephony)
    MULAW_8000 = "mulaw_8000"  # G.711 mu-law 8kHz (Twilio/Vonage)
    ALAW_8000 = "alaw_8000"  # G.711 A-law 8kHz (European telephony)


# Format metadata dictionary
AUDIO_FORMAT_INFO: Dict[AudioFormat, Dict[str, Any]] = {
    AudioFormat.WAV: {
        "sample_rate": 24000,
        "bits_per_sample": 16,
        "encoding": "pcm_signed",
        "channels": 1,
        "media_type": "audio/wav",
        "description": "Standard WAV file with header",
        "use_case": "General purpose, file storage",
    },
    AudioFormat.PCM_24000_16: {
        "sample_rate": 24000,
        "bits_per_sample": 16,
        "encoding": "pcm_signed",
        "channels": 1,
        "media_type": "audio/pcm",
        "description": "Raw PCM at native sample rate",
        "use_case": "Low-latency streaming, custom players",
    },
    AudioFormat.PCM_24000_F32: {
        "sample_rate": 24000,
        "bits_per_sample": 32,
        "encoding": "pcm_float",
        "channels": 1,
        "media_type": "audio/pcm",
        "description": "32-bit float PCM at native rate",
        "use_case": "Audio processing pipelines",
    },
    AudioFormat.PCM_16000_16: {
        "sample_rate": 16000,
        "bits_per_sample": 16,
        "encoding": "pcm_signed",
        "channels": 1,
        "media_type": "audio/pcm",
        "description": "16kHz PCM for speech recognition",
        "use_case": "ASR systems (Whisper, DeepSpeech)",
    },
    AudioFormat.PCM_8000_16: {
        "sample_rate": 8000,
        "bits_per_sample": 16,
        "encoding": "pcm_signed",
        "channels": 1,
        "media_type": "audio/pcm",
        "description": "8kHz PCM for telephony",
        "use_case": "VoIP, telephony systems",
    },
    AudioFormat.MULAW_8000: {
        "sample_rate": 8000,
        "bits_per_sample": 8,
        "encoding": "mulaw",
        "channels": 1,
        "media_type": "audio/basic",
        "description": "G.711 mu-law (North America/Japan)",
        "use_case": "Twilio, Vonage, US telephony",
    },
    AudioFormat.ALAW_8000: {
        "sample_rate": 8000,
        "bits_per_sample": 8,
        "encoding": "alaw",
        "channels": 1,
        "media_type": "audio/basic",
        "description": "G.711 A-law (Europe/International)",
        "use_case": "European telephony, ISDN",
    },
}


# G.711 mu-law encoding constants (ITU-T standard)
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635

# Mu-law segment encoding table
_MULAW_ENCODE_TABLE = np.array(
    [
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ],
    dtype=np.uint8,
)


def _lin2mulaw(samples: np.ndarray) -> np.ndarray:
    """
    Convert 16-bit linear PCM samples to mu-law (ITU-T G.711).

    Args:
        samples: Int16 audio samples

    Returns:
        Uint8 mu-law encoded samples
    """
    # Ensure int16 and work with int32 for calculations
    samples = np.clip(samples, -32768, 32767).astype(np.int32)

    # Extract sign bit
    sign = (samples >> 8) & 0x80

    # Get magnitude (absolute value)
    samples = np.where(samples < 0, -samples, samples)

    # Clip to max value
    samples = np.clip(samples, 0, _MULAW_CLIP)

    # Add bias
    samples = samples + _MULAW_BIAS

    # Get exponent from lookup table
    exponent = _MULAW_ENCODE_TABLE[(samples >> 7) & 0xFF]

    # Get mantissa
    mantissa = (samples >> (exponent + 3)) & 0x0F

    # Combine and complement
    mulaw = ~(sign | (exponent.astype(np.int32) << 4) | mantissa) & 0xFF

    return mulaw.astype(np.uint8)


def _lin2alaw(samples: np.ndarray) -> np.ndarray:
    """
    Convert 16-bit linear PCM samples to A-law (ITU-T G.711).

    Args:
        samples: Int16 audio samples

    Returns:
        Uint8 A-law encoded samples
    """
    # Ensure int16 and work with int32 for calculations
    samples = np.clip(samples, -32768, 32767).astype(np.int32)

    # Extract sign (inverted for A-law)
    sign = ((~samples) >> 8) & 0x80

    # Get magnitude
    samples = np.abs(samples)

    # Initialize result
    result = np.zeros(len(samples), dtype=np.uint8)

    # Process samples >= 256 (logarithmic region)
    mask_large = samples >= 256
    if np.any(mask_large):
        s = samples[mask_large]
        # Find exponent by checking bit positions
        exponent = np.zeros(len(s), dtype=np.int32)
        for i in range(7, 0, -1):
            threshold = 1 << (i + 8)
            exponent = np.where((s >= threshold) & (exponent == 0), i, exponent)
        # Adjust for samples that didn't match any threshold
        exponent = np.where(
            (exponent == 0) & (s >= 256),
            np.floor(np.log2(np.maximum(s, 1)) - 7).astype(np.int32),
            exponent,
        )
        exponent = np.clip(exponent, 1, 7)
        mantissa = (s >> (exponent + 3)) & 0x0F
        result[mask_large] = ((exponent << 4) | mantissa).astype(np.uint8)

    # Process samples < 256 (linear region)
    mask_small = samples < 256
    if np.any(mask_small):
        result[mask_small] = (samples[mask_small] >> 4).astype(np.uint8)

    # Apply sign and XOR pattern
    return ((sign | result) ^ 0x55).astype(np.uint8)


def _safe_float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Safely convert float32 audio to int16, handling NaN/Inf.

    Args:
        audio: Float32 audio array (expected range -1.0 to 1.0)

    Returns:
        Int16 audio array
    """
    # Handle NaN and Inf
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    # Clip and convert
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio using linear interpolation.

    For production use with quality requirements, consider using
    scipy.signal.resample_poly or librosa.resample instead.

    Args:
        audio: Input audio samples (float32)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Calculate output length
    duration = len(audio) / orig_sr
    output_len = int(duration * target_sr)

    if output_len == 0:
        return np.array([], dtype=audio.dtype)

    # Linear interpolation
    x_orig = np.arange(len(audio))
    x_new = np.linspace(0, len(audio) - 1, output_len)

    return np.interp(x_new, x_orig, audio).astype(audio.dtype)


def convert_audio_format(
    audio: np.ndarray,
    source_sr: int,
    target_format: AudioFormat,
) -> bytes:
    """
    Convert audio to the specified output format.

    Args:
        audio: Float32 audio array from model (typically 24kHz)
        source_sr: Source sample rate (typically 24000)
        target_format: Target AudioFormat

    Returns:
        Bytes in the target format
    """
    format_info = AUDIO_FORMAT_INFO[target_format]
    target_sr = format_info["sample_rate"]
    encoding = format_info["encoding"]

    # Resample if needed
    if source_sr != target_sr:
        audio = resample_audio(audio, source_sr, target_sr)

    # Convert based on encoding
    if encoding == "pcm_float":
        # Float32 PCM
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio.tobytes()

    elif encoding == "pcm_signed":
        # Int16 PCM
        audio_int16 = _safe_float32_to_int16(audio)
        return audio_int16.tobytes()

    elif encoding == "mulaw":
        # Convert to int16, then mu-law
        audio_int16 = _safe_float32_to_int16(audio)
        mulaw = _lin2mulaw(audio_int16)
        return mulaw.tobytes()

    elif encoding == "alaw":
        # Convert to int16, then A-law
        audio_int16 = _safe_float32_to_int16(audio)
        alaw = _lin2alaw(audio_int16)
        return alaw.tobytes()

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert audio array to complete WAV file bytes.

    Args:
        audio: Float32 audio array
        sample_rate: Sample rate
        bits_per_sample: 16 or 32

    Returns:
        Complete WAV file as bytes
    """
    # Convert to appropriate format
    if bits_per_sample == 16:
        audio_data = _safe_float32_to_int16(audio)
        audio_format = 1  # PCM
    elif bits_per_sample == 32:
        audio_data = np.clip(audio, -1.0, 1.0).astype(np.float32)
        audio_format = 3  # IEEE float
    else:
        raise ValueError(f"Unsupported bits_per_sample: {bits_per_sample}")

    num_channels = 1
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = len(audio_data) * bytes_per_sample

    buffer = io.BytesIO()

    # RIFF header
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + data_size))
    buffer.write(b"WAVE")

    # fmt chunk
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<I", 16))  # chunk size
    buffer.write(struct.pack("<H", audio_format))
    buffer.write(struct.pack("<H", num_channels))
    buffer.write(struct.pack("<I", sample_rate))
    buffer.write(struct.pack("<I", byte_rate))
    buffer.write(struct.pack("<H", block_align))
    buffer.write(struct.pack("<H", bits_per_sample))

    # data chunk
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(audio_data.tobytes())

    return buffer.getvalue()


def get_audio_format_headers(
    audio_format: AudioFormat,
    audio_duration_ms: Optional[float] = None,
    generation_time_ms: Optional[float] = None,
) -> dict:
    """
    Get HTTP headers for audio response.

    Args:
        audio_format: Audio format
        audio_duration_ms: Duration of audio in ms
        generation_time_ms: Time to generate in ms

    Returns:
        Dictionary of headers
    """
    info = AUDIO_FORMAT_INFO[audio_format]

    headers = {
        "X-Audio-Format": audio_format.value,
        "X-Audio-Sample-Rate": str(info["sample_rate"]),
        "X-Audio-Bits-Per-Sample": str(info["bits_per_sample"]),
        "X-Audio-Encoding": info["encoding"],
        "X-Audio-Channels": str(info["channels"]),
    }

    if audio_duration_ms is not None:
        headers["X-Audio-Duration-Ms"] = str(round(audio_duration_ms, 2))

    if generation_time_ms is not None:
        headers["X-Generation-Time-Ms"] = str(round(generation_time_ms, 2))
        if audio_duration_ms and audio_duration_ms > 0:
            rtf = generation_time_ms / audio_duration_ms
            headers["X-Real-Time-Factor"] = str(round(rtf, 3))

    return headers


def get_format_info_dict(audio_format: AudioFormat) -> dict:
    """
    Get format info as a dictionary for JSON serialization.

    Args:
        audio_format: Audio format

    Returns:
        Dictionary with format metadata
    """
    info = AUDIO_FORMAT_INFO[audio_format]
    return {
        "sample_rate": info["sample_rate"],
        "bits_per_sample": info["bits_per_sample"],
        "encoding": info["encoding"],
        "channels": info["channels"],
    }


# Voice management constants
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MIN_AUDIO_FILE_SIZE = 1024  # 1KB minimum
MAX_AUDIO_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximum
