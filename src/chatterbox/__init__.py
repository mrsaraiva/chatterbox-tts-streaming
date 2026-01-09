try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .tts import ChatterboxTTS
from .tts_turbo import ChatterboxTurboTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .streaming import StreamingMetrics, S3GenStreamer

__all__ = [
    "ChatterboxTTS",
    "ChatterboxTurboTTS",
    "ChatterboxVC",
    "ChatterboxMultilingualTTS",
    "SUPPORTED_LANGUAGES",
    "StreamingMetrics",
    "S3GenStreamer",
]