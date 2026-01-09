"""S3Gen streaming wrapper for incremental audio generation."""

import logging
from typing import Optional, List, Tuple
import torch

from ..models.s3gen import S3Gen, S3GEN_SR
from ..models.s3gen.const import S3GEN_SIL

logger = logging.getLogger(__name__)


class S3GenStreamer:
    """
    Streaming wrapper for S3Gen that buffers tokens and generates audio chunks.

    This class handles:
    - Token buffering until chunk_size is reached
    - Incremental mel-spectrogram generation with finalize=False
    - Stateful HiFiGAN vocoding with cache_source
    - Final chunk processing with finalize=True

    Args:
        s3gen: The S3Gen model instance
        ref_dict: Reference voice embeddings from s3gen.embed_ref()
        chunk_size: Number of tokens per audio chunk (default: 50)
        n_cfm_timesteps: CFM denoising steps (default: 2 for meanflow, 10 otherwise)
    """

    # Minimum tokens needed before we can generate audio (pre_lookahead_len)
    MIN_TOKENS = 3

    def __init__(
        self,
        s3gen: S3Gen,
        ref_dict: dict,
        chunk_size: int = 50,
        n_cfm_timesteps: Optional[int] = None,
    ):
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.chunk_size = max(chunk_size, self.MIN_TOKENS + 1)  # Ensure valid chunk size
        self.n_cfm_timesteps = n_cfm_timesteps or (2 if s3gen.meanflow else 10)

        # Token buffer
        self.token_buffer: List[int] = []
        self.processed_tokens: int = 0

        # HiFiGAN cache for continuous audio
        self.hift_cache: Optional[torch.Tensor] = None

        # Track if we've generated any audio
        self.has_generated: bool = False

    @property
    def device(self):
        return self.s3gen.device

    @property
    def dtype(self):
        return self.s3gen.dtype

    def reset(self):
        """Reset the streamer state for a new generation."""
        self.token_buffer = []
        self.processed_tokens = 0
        self.hift_cache = None
        self.has_generated = False

    def add_token(self, token: int) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Add a token to the buffer and return audio if chunk is ready.

        Args:
            token: Speech token to add

        Returns:
            Tuple of (audio_chunk, num_tokens) if chunk is ready, None otherwise.
            audio_chunk is a tensor of shape (num_samples,) at 24kHz.
        """
        self.token_buffer.append(token)

        # Check if we have enough tokens for a chunk
        if len(self.token_buffer) >= self.chunk_size:
            return self._generate_chunk(finalize=False)

        return None

    def flush(self) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Process any remaining tokens in the buffer.

        This should be called after all tokens have been added to generate
        the final audio chunk with finalize=True.

        Returns:
            Tuple of (audio_chunk, num_tokens) if there are remaining tokens,
            None if buffer is empty.
        """
        if len(self.token_buffer) == 0:
            return None

        # Add silence tokens at the end for clean finish
        self.token_buffer.extend([S3GEN_SIL] * 3)

        return self._generate_chunk(finalize=True)

    def _generate_chunk(self, finalize: bool) -> Tuple[torch.Tensor, int]:
        """
        Generate audio from the current token buffer.

        Args:
            finalize: If True, process all remaining tokens. If False, leave
                     pre_lookahead tokens for the next chunk.

        Returns:
            Tuple of (audio_chunk, num_tokens_processed)
        """
        # Convert tokens to tensor
        tokens = torch.tensor(self.token_buffer, dtype=torch.long, device=self.device)
        tokens = tokens.unsqueeze(0)  # Add batch dimension

        # Generate mel spectrogram
        with torch.inference_mode():
            output_mels = self.s3gen.flow_inference(
                speech_tokens=tokens,
                ref_dict=self.ref_dict,
                n_cfm_timesteps=self.n_cfm_timesteps,
                finalize=finalize,
            )

            # Convert mels to audio using HiFiGAN
            if self.hift_cache is None:
                self.hift_cache = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype)

            output_wav, self.hift_cache = self.s3gen.hift_inference(
                speech_feat=output_mels,
                cache_source=self.hift_cache,
            )

        # Calculate how many tokens were actually processed
        if finalize:
            num_tokens = len(self.token_buffer)
            self.token_buffer = []
        else:
            # When not finalizing, S3Gen drops the last 3 tokens (pre_lookahead)
            # So we keep those in the buffer for the next chunk
            num_tokens = len(self.token_buffer) - self.MIN_TOKENS
            self.token_buffer = self.token_buffer[-self.MIN_TOKENS:]

        self.processed_tokens += num_tokens
        self.has_generated = True

        # Return audio without batch dimension
        audio = output_wav.squeeze(0)

        # Apply fade-in on first chunk to reduce artifacts
        if not self.has_generated:
            n_trim = S3GEN_SR // 50  # 20ms
            if len(audio) > 2 * n_trim:
                fade = torch.zeros(2 * n_trim, device=audio.device, dtype=audio.dtype)
                fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim, device=audio.device)) + 1) / 2
                audio[:2 * n_trim] *= fade

        return audio, num_tokens

    def get_pending_tokens(self) -> int:
        """Return the number of tokens waiting in the buffer."""
        return len(self.token_buffer)

    def get_processed_tokens(self) -> int:
        """Return the total number of tokens that have been converted to audio."""
        return self.processed_tokens
