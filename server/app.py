"""FastAPI application for Chatterbox TTS API."""

import asyncio
import base64
import io
import json
import logging
import os
import struct
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

from .model_manager import ModelManager, model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    model_type = os.getenv("CHATTERBOX_MODEL", "turbo")
    device = os.getenv("CHATTERBOX_DEVICE", None)
    await model_manager.initialize(model_type=model_type, device=device)
    yield


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


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    # Normalize and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV header
    num_samples = len(audio_int16)
    bytes_per_sample = 2
    num_channels = 1
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample

    buffer = io.BytesIO()
    # RIFF header
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + data_size))
    buffer.write(b"WAVE")
    # fmt chunk
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<I", 16))  # chunk size
    buffer.write(struct.pack("<H", 1))   # PCM format
    buffer.write(struct.pack("<H", num_channels))
    buffer.write(struct.pack("<I", sample_rate))
    buffer.write(struct.pack("<I", byte_rate))
    buffer.write(struct.pack("<H", block_align))
    buffer.write(struct.pack("<H", 16))  # bits per sample
    # data chunk
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(audio_int16.tobytes())

    return buffer.getvalue()


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
) -> Response:
    """
    Generate speech audio (returns complete WAV file).

    This is a synchronous endpoint that waits for full generation.
    For real-time streaming, use the WebSocket or SSE endpoints.

    For multilingual model, the `language` parameter is required.
    Supported languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
    """
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

    # Convert to WAV
    audio_np = audio.squeeze().cpu().numpy()
    wav_bytes = audio_to_wav_bytes(audio_np, model_manager.model.sr)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.get("/v1/tts/stream-sse")
async def stream_speech_sse(
    text: str = Query(..., description="Text to synthesize"),
    voice_id: str = Query("default", description="Voice ID to use"),
    chunk_size: int = Query(50, description="Tokens per chunk (default: 50)"),
    temperature: float = Query(0.8, description="Sampling temperature"),
) -> StreamingResponse:
    """
    Server-Sent Events streaming TTS.

    Returns an event stream with base64-encoded audio chunks.
    Each event contains either audio data or metrics.

    Events:
    - `audio`: Base64-encoded PCM audio (int16, 24kHz, mono)
    - `metrics`: JSON metrics for the chunk
    - `done`: Generation complete
    """
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        await model_manager.set_voice(voice_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    async def generate():
        async with model_manager.request_lock:
            loop = asyncio.get_event_loop()

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

            while True:
                result = await loop.run_in_executor(None, get_next_chunk, stream)
                if result is None:
                    break

                audio_chunk, metrics = result
                audio_np = audio_chunk.cpu().numpy()

                # Convert to int16 PCM
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()

                # Send audio event
                yield f"event: audio\ndata: {audio_b64}\n\n"

                # Send metrics event
                yield f"event: metrics\ndata: {json.dumps(metrics.to_dict())}\n\n"

        # Send done event
        yield "event: done\ndata: {}\n\n"

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
    1. Client sends JSON: {"action": "generate", "text": "...", "voice_id": "default", ...}
    2. Server sends binary audio chunks (PCM int16, 24kHz, mono)
    3. Server sends JSON metrics after each chunk
    4. Server sends JSON {"type": "done", ...} when complete

    Client can also send:
    - {"action": "ping"} - Server responds with {"type": "pong"}
    - {"action": "stop"} - Cancel current generation
    """
    await websocket.accept()

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action == "generate":
                await handle_generate(websocket, message)
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


async def handle_generate(websocket: WebSocket, message: dict):
    """Handle a generate request over WebSocket."""
    text = message.get("text", "")
    voice_id = message.get("voice_id", "default")
    chunk_size = message.get("chunk_size", 50)
    temperature = message.get("temperature", 0.8)

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

    # Send start message
    await websocket.send_json({"type": "start", "text": text, "voice_id": voice_id})

    async with model_manager.request_lock:
        loop = asyncio.get_event_loop()

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
            result = await loop.run_in_executor(None, get_next_chunk, stream)
            if result is None:
                break

            audio_chunk, metrics = result
            audio_np = audio_chunk.cpu().numpy()
            chunk_count += 1
            total_samples += len(audio_np)

            # Convert to int16 PCM and send as binary
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            await websocket.send_bytes(audio_int16.tobytes())

            # Send metrics as JSON
            await websocket.send_json({
                "type": "metrics",
                "chunk": chunk_count,
                **metrics.to_dict(),
            })

    # Send completion message
    total_time = (time.perf_counter() - start_time) * 1000
    audio_duration = (total_samples / model_manager.model.sr) * 1000

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
            }
            for voice_id, config in voices.items()
        ]
    }


@app.post("/v1/voices/create")
async def create_voice(
    voice_id: str = Form(..., description="Unique voice identifier"),
    name: str = Form(None, description="Display name for the voice"),
    audio_file: UploadFile = File(..., description="Reference audio (6-15 seconds)"),
):
    """
    Create a new voice from reference audio.

    The audio file should be 6-15 seconds of clear speech from the target voice.
    """
    if voice_id in model_manager.voices:
        raise HTTPException(status_code=409, detail=f"Voice already exists: {voice_id}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        config = await model_manager.create_voice(
            voice_id=voice_id,
            audio_path=tmp_path,
            name=name,
        )
        return {
            "id": voice_id,
            "name": config.name,
            "message": "Voice created successfully",
        }
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
