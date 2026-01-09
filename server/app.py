"""FastAPI application for Chatterbox TTS API."""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .audio import (
    AudioFormat,
    AUDIO_FORMAT_INFO,
    convert_audio_format,
    audio_to_wav_bytes,
    get_audio_format_headers,
    get_format_info_dict,
    ALLOWED_AUDIO_EXTENSIONS,
    MIN_AUDIO_FILE_SIZE,
    MAX_AUDIO_FILE_SIZE,
)
from .model_manager import ModelManager, model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup, cleanup on shutdown."""
    model_type = os.getenv("CHATTERBOX_MODEL", "turbo")
    device = os.getenv("CHATTERBOX_DEVICE", None)
    await model_manager.initialize(model_type=model_type, device=device)
    yield
    # Cleanup on shutdown
    await model_manager.shutdown()


app = FastAPI(
    title="Chatterbox TTS API",
    description="Real-time streaming text-to-speech API powered by Chatterbox",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "model_type": model_manager.model_type,
        "device": model_manager.device,
        "voices_count": len(model_manager.voices),
    }


@app.get("/v1/formats")
async def list_formats():
    """
    List all supported audio output formats.

    Returns format metadata including sample rate, encoding,
    and recommended use cases.
    """
    return {
        "formats": [
            {
                "format": fmt.value,
                **info,
            }
            for fmt, info in AUDIO_FORMAT_INFO.items()
        ],
        "default": AudioFormat.WAV.value,
        "native_sample_rate": 24000,
    }


@app.get("/v1/languages")
async def list_languages():
    """List supported languages (for multilingual model only)."""
    if model_manager.model_type != "multilingual":
        return {"languages": [], "note": "Language selection only available with multilingual model"}

    from chatterbox import SUPPORTED_LANGUAGES
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in SUPPORTED_LANGUAGES.items()
        ]
    }


@app.post("/v1/tts/generate")
async def generate_speech(
    text: str = Form(..., description="Text to synthesize"),
    voice_id: str = Form("default", description="Voice ID to use"),
    language: str = Form(None, description="Language code (e.g., 'en', 'es', 'fr') - required for multilingual model"),
    temperature: float = Form(0.8, description="Sampling temperature"),
    exaggeration: float = Form(0.5, description="Emotion exaggeration (0.0-1.0)"),
    output_format: str = Form("wav", description="Output audio format (wav, pcm_24000_16, mulaw_8000, etc.)"),
) -> Response:
    """
    Generate speech audio.

    Supports multiple output formats:
    - wav: Complete WAV file (default)
    - pcm_24000_16: Raw 24kHz 16-bit PCM
    - pcm_24000_f32: Raw 24kHz 32-bit float PCM
    - pcm_16000_16: 16kHz 16-bit PCM (for speech recognition)
    - pcm_8000_16: 8kHz 16-bit PCM (for telephony)
    - mulaw_8000: G.711 mu-law 8kHz (Twilio/Vonage)
    - alaw_8000: G.711 A-law 8kHz (European telephony)

    For multilingual model, the `language` parameter is required.
    """
    # Validate output format
    try:
        audio_format = AudioFormat(output_format.lower())
    except ValueError:
        valid_formats = [f.value for f in AudioFormat]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format '{output_format}'. Valid: {valid_formats}"
        )

    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate language for multilingual model
    if model_manager.model_type == "multilingual":
        if not language:
            raise HTTPException(status_code=400, detail="Language parameter required for multilingual model")
        from chatterbox import SUPPORTED_LANGUAGES
        if language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )

    # Set voice
    try:
        await model_manager.set_voice(voice_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Generate audio
    start_time = time.perf_counter()

    async with model_manager.request_lock:
        loop = asyncio.get_event_loop()

        def generate():
            if model_manager.model_type == "multilingual":
                return model_manager.model.generate(
                    text=text,
                    language_id=language,
                    temperature=temperature,
                    exaggeration=exaggeration,
                )
            else:
                return model_manager.model.generate(
                    text=text,
                    temperature=temperature,
                    exaggeration=exaggeration,
                )

        audio = await loop.run_in_executor(None, generate)

    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Convert to numpy
    audio_np = audio.squeeze().cpu().numpy()
    sample_rate = model_manager.model.sr
    audio_duration_ms = (len(audio_np) / sample_rate) * 1000

    # Convert to requested format
    if audio_format == AudioFormat.WAV:
        content = audio_to_wav_bytes(audio_np, sample_rate)
        filename = "speech.wav"
    else:
        content = convert_audio_format(audio_np, sample_rate, audio_format)
        ext = audio_format.value.replace("_", ".")
        filename = f"speech.{ext}"

    # Get format info and headers
    format_info = AUDIO_FORMAT_INFO[audio_format]
    headers = get_audio_format_headers(
        audio_format,
        audio_duration_ms=audio_duration_ms,
        generation_time_ms=generation_time_ms,
    )
    headers["Content-Disposition"] = f"attachment; filename={filename}"

    return Response(
        content=content,
        media_type=format_info["media_type"],
        headers=headers,
    )


@app.get("/v1/tts/stream-sse")
async def stream_speech_sse(
    text: str = Query(..., description="Text to synthesize"),
    voice_id: str = Query("default", description="Voice ID to use"),
    chunk_size: int = Query(50, description="Tokens per chunk (default: 50)"),
    temperature: float = Query(0.8, description="Sampling temperature"),
    output_format: str = Query("pcm_24000_16", description="Output audio format"),
    language: str = Query(None, description="Language code for multilingual model"),
) -> StreamingResponse:
    """
    Server-Sent Events streaming TTS.

    Returns an event stream with base64-encoded audio chunks.

    Events:
    - `format`: Audio format metadata (sent first)
    - `audio`: Base64-encoded audio chunk with index
    - `metrics`: JSON metrics for the chunk
    - `done`: Generation complete
    - `error`: Error occurred

    Supported output formats: pcm_24000_16, pcm_16000_16, mulaw_8000, alaw_8000, etc.
    Note: WAV format is not suitable for streaming; use pcm_24000_16 instead.
    """
    # Validate output format
    try:
        audio_format = AudioFormat(output_format.lower())
        if audio_format == AudioFormat.WAV:
            audio_format = AudioFormat.PCM_24000_16  # WAV not suitable for streaming
    except ValueError:
        valid_formats = [f.value for f in AudioFormat if f != AudioFormat.WAV]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format. Valid for streaming: {valid_formats}"
        )

    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate language for multilingual model
    if model_manager.model_type == "multilingual":
        if not language:
            raise HTTPException(status_code=400, detail="Language parameter required for multilingual model")
        from chatterbox import SUPPORTED_LANGUAGES
        if language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'"
            )

    try:
        await model_manager.set_voice(voice_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    async def generate():
        try:
            # Send format info first
            format_data = json.dumps(get_format_info_dict(audio_format))
            yield f"event: format\ndata: {format_data}\n\n"

            async with model_manager.request_lock:
                loop = asyncio.get_event_loop()
                source_sr = model_manager.model.sr

                # Create generator in thread
                def create_stream():
                    return model_manager.model.generate_stream(
                        text=text,
                        chunk_size=chunk_size,
                        temperature=temperature,
                    )

                stream = await loop.run_in_executor(None, create_stream)

                # Yield chunks
                def get_next_chunk(gen):
                    try:
                        return next(gen)
                    except StopIteration:
                        return None

                chunk_index = 0
                while True:
                    result = await loop.run_in_executor(None, get_next_chunk, stream)
                    if result is None:
                        break

                    audio_chunk, metrics = result
                    audio_np = audio_chunk.cpu().numpy()

                    # Convert to requested format
                    audio_bytes = convert_audio_format(audio_np, source_sr, audio_format)
                    audio_b64 = base64.b64encode(audio_bytes).decode()

                    # Send audio event with index
                    audio_data = json.dumps({
                        "chunk": audio_b64,
                        "index": chunk_index,
                    })
                    yield f"event: audio\ndata: {audio_data}\n\n"

                    # Send metrics event
                    yield f"event: metrics\ndata: {json.dumps(metrics.to_dict())}\n\n"

                    chunk_index += 1

            # Send done event with status
            done_data = json.dumps({"status": "complete", "total_chunks": chunk_index})
            yield f"event: done\ndata: {done_data}\n\n"

        except Exception as e:
            # Send error event
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"
            logger.error(f"SSE streaming error: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/v1/tts/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket streaming TTS.

    Protocol:
    1. Client sends JSON: {"action": "generate", "text": "...", "voice_id": "default", "output_format": "pcm_24000_16", ...}
    2. Server sends JSON: {"type": "format", "data": {sample_rate, bits, encoding, channels}}
    3. Server sends JSON: {"type": "start", ...}
    4. Server sends binary audio chunks (in requested format)
    5. Server sends JSON metrics after each chunk
    6. Server sends JSON {"type": "done", ...} when complete

    Client can also send:
    - {"action": "ping"} - Server responds with {"type": "pong"}
    - {"action": "stop"} - Cancel current generation, server responds with {"type": "stopped"}

    Supported output_format values:
    - pcm_24000_16: Raw 24kHz 16-bit PCM (default)
    - pcm_16000_16: 16kHz 16-bit PCM
    - pcm_8000_16: 8kHz 16-bit PCM
    - mulaw_8000: G.711 mu-law 8kHz (telephony)
    - alaw_8000: G.711 A-law 8kHz (telephony)
    """
    await websocket.accept()

    # Track active generation for cancellation
    stop_flag = asyncio.Event()
    is_generating = False

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action == "stop":
                if is_generating:
                    stop_flag.set()
                    # The generate handler will send "stopped" when it exits
                else:
                    await websocket.send_json({"type": "stopped", "message": "No active generation"})
                continue

            if action == "generate":
                stop_flag.clear()
                is_generating = True
                try:
                    await handle_generate(websocket, message, stop_flag)
                finally:
                    is_generating = False
                continue

            await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


async def handle_generate(websocket: WebSocket, message: dict, stop_flag: asyncio.Event):
    """Handle a generate request over WebSocket."""
    text = message.get("text", "")
    voice_id = message.get("voice_id", "default")
    chunk_size = message.get("chunk_size", 50)
    temperature = message.get("temperature", 0.8)
    output_format_str = message.get("output_format", "pcm_24000_16")

    # Validate output format
    try:
        output_format = AudioFormat(output_format_str.lower())
        if output_format == AudioFormat.WAV:
            output_format = AudioFormat.PCM_24000_16  # WAV not suitable for streaming
    except ValueError:
        valid_formats = [f.value for f in AudioFormat if f != AudioFormat.WAV]
        await websocket.send_json({
            "type": "error",
            "message": f"Invalid output_format. Valid: {valid_formats}"
        })
        return

    if not text:
        await websocket.send_json({"type": "error", "message": "No text provided"})
        return

    if model_manager.model is None:
        await websocket.send_json({"type": "error", "message": "Model not loaded"})
        return

    try:
        await model_manager.set_voice(voice_id)
    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return

    # Send format info first
    await websocket.send_json({
        "type": "format",
        "data": get_format_info_dict(output_format)
    })

    # Send start message
    await websocket.send_json({
        "type": "start",
        "text": text,
        "voice_id": voice_id,
        "output_format": output_format.value,
    })

    async with model_manager.request_lock:
        loop = asyncio.get_event_loop()
        source_sr = model_manager.model.sr

        def create_stream():
            return model_manager.model.generate_stream(
                text=text,
                chunk_size=chunk_size,
                temperature=temperature,
            )

        stream = await loop.run_in_executor(None, create_stream)

        def get_next_chunk(gen):
            try:
                return next(gen)
            except StopIteration:
                return None

        chunk_count = 0
        total_samples = 0
        start_time = time.perf_counter()

        while True:
            # Check for stop signal
            if stop_flag.is_set():
                await websocket.send_json({
                    "type": "stopped",
                    "chunks_generated": chunk_count,
                    "samples_generated": total_samples,
                })
                return

            result = await loop.run_in_executor(None, get_next_chunk, stream)
            if result is None:
                break

            audio_chunk, metrics = result
            audio_np = audio_chunk.cpu().numpy()
            chunk_count += 1
            total_samples += len(audio_np)

            # Convert to requested format
            audio_bytes = convert_audio_format(audio_np, source_sr, output_format)
            await websocket.send_bytes(audio_bytes)

            # Send metrics as JSON
            await websocket.send_json({
                "type": "metrics",
                "chunk": chunk_count,
                **metrics.to_dict(),
            })

    # Send completion message
    total_time = (time.perf_counter() - start_time) * 1000
    audio_duration = (total_samples / source_sr) * 1000

    await websocket.send_json({
        "type": "done",
        "total_chunks": chunk_count,
        "total_samples": total_samples,
        "total_latency_ms": round(total_time, 2),
        "audio_duration_ms": round(audio_duration, 2),
        "final_rtf": round(total_time / audio_duration, 3) if audio_duration > 0 else 0,
    })


# ============================================================================
# Voice Management Endpoints
# ============================================================================

@app.get("/v1/voices")
async def list_voices():
    """List all available voices."""
    voices = model_manager.list_voices()
    return {
        "voices": [
            {
                "id": voice_id,
                "name": config.name,
                "created_at": config.created_at,
                "language": config.language,
                "is_default": config.is_default,
            }
            for voice_id, config in voices.items()
        ]
    }


@app.post("/v1/voices/create")
async def create_voice(
    voice_id: str = Form(..., description="Unique voice identifier"),
    name: str = Form(None, description="Display name for the voice"),
    language: str = Form("en", description="Language code (e.g., 'en', 'es', 'fr')"),
    audio_file: UploadFile = File(..., description="Reference audio (6-15 seconds)"),
):
    """
    Create a new voice from reference audio.

    The audio file should be 6-15 seconds of clear speech.
    Supported formats: WAV, MP3, FLAC, OGG (max 50MB).
    """
    if voice_id in model_manager.voices:
        raise HTTPException(status_code=409, detail=f"Voice already exists: {voice_id}")

    # Validate file type
    filename = audio_file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # Read and validate file size
    content = await audio_file.read()
    if len(content) < MIN_AUDIO_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too small (minimum {MIN_AUDIO_FILE_SIZE} bytes)")
    if len(content) > MAX_AUDIO_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large (maximum {MAX_AUDIO_FILE_SIZE // (1024*1024)}MB)")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        config = await model_manager.create_voice(
            voice_id=voice_id,
            audio_path=tmp_path,
            name=name,
            language=language,
        )
        return {
            "id": voice_id,
            "name": config.name,
            "language": config.language,
            "message": "Voice created successfully",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a custom voice."""
    try:
        deleted = await model_manager.delete_voice(voice_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
        return {"message": f"Voice deleted: {voice_id}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/voices/{voice_id}/set")
async def set_current_voice(voice_id: str):
    """
    Set the current active voice.

    This voice will be used for subsequent TTS requests
    if no voice_id is specified.
    """
    try:
        await model_manager.set_voice(voice_id)
        return {
            "message": f"Current voice set to: {voice_id}",
            "voice_id": voice_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
