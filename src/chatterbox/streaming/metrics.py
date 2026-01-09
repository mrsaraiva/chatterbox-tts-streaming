"""Streaming metrics for tracking TTS generation performance."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class StreamingMetrics:
    """
    Metrics for streaming TTS generation.

    Attributes:
        first_chunk_latency_ms: Time from request to first audio chunk (ms)
        chunk_latency_ms: Time to generate the current chunk (ms)
        total_latency_ms: Cumulative generation time (ms)
        real_time_factor: RTF = processing_time / audio_duration (< 1.0 is faster than real-time)
        tokens_generated: Total speech tokens generated so far
        chunks_generated: Number of audio chunks yielded
        audio_duration_ms: Total audio duration generated so far (ms)
    """
    first_chunk_latency_ms: float = 0.0
    chunk_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    real_time_factor: float = 0.0
    tokens_generated: int = 0
    chunks_generated: int = 0
    audio_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "first_chunk_latency_ms": round(self.first_chunk_latency_ms, 2),
            "chunk_latency_ms": round(self.chunk_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "real_time_factor": round(self.real_time_factor, 3),
            "tokens_generated": self.tokens_generated,
            "chunks_generated": self.chunks_generated,
            "audio_duration_ms": round(self.audio_duration_ms, 2),
        }

    def __str__(self) -> str:
        return (
            f"Chunk {self.chunks_generated}: "
            f"latency={self.chunk_latency_ms:.0f}ms, "
            f"RTF={self.real_time_factor:.3f}, "
            f"tokens={self.tokens_generated}"
        )


class StreamingTimer:
    """Helper class to track timing for streaming metrics."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.chunk_start_time: Optional[float] = None
        self.total_audio_samples: int = 0
        self.sample_rate: int = 24000  # S3GEN_SR
        self.tokens_generated: int = 0
        self.chunks_generated: int = 0

    def start(self):
        """Start the overall timer."""
        self.start_time = time.perf_counter()
        self.chunk_start_time = self.start_time

    def start_chunk(self):
        """Mark the start of a new chunk."""
        self.chunk_start_time = time.perf_counter()

    def record_chunk(self, audio_samples: int, tokens: int) -> StreamingMetrics:
        """
        Record metrics for a completed chunk.

        Args:
            audio_samples: Number of audio samples in this chunk
            tokens: Number of tokens processed for this chunk

        Returns:
            StreamingMetrics for this chunk
        """
        now = time.perf_counter()

        # Track first chunk
        if self.first_chunk_time is None:
            self.first_chunk_time = now

        # Update counters
        self.total_audio_samples += audio_samples
        self.tokens_generated += tokens
        self.chunks_generated += 1

        # Calculate times
        chunk_latency_ms = (now - self.chunk_start_time) * 1000
        total_latency_ms = (now - self.start_time) * 1000
        first_chunk_latency_ms = (self.first_chunk_time - self.start_time) * 1000

        # Calculate audio duration and RTF
        audio_duration_ms = (self.total_audio_samples / self.sample_rate) * 1000
        real_time_factor = total_latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0.0

        return StreamingMetrics(
            first_chunk_latency_ms=first_chunk_latency_ms,
            chunk_latency_ms=chunk_latency_ms,
            total_latency_ms=total_latency_ms,
            real_time_factor=real_time_factor,
            tokens_generated=self.tokens_generated,
            chunks_generated=self.chunks_generated,
            audio_duration_ms=audio_duration_ms,
        )
