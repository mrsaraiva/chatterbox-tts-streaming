#!/usr/bin/env python3
"""
Example: Streaming TTS with Real-Time Playback

This example demonstrates how to use the streaming API to generate
speech in real-time, yielding audio chunks as they become available.

Usage:
    python example_tts_stream.py [--device cuda] [--chunk-size 50] [--text "Hello world"]

Requirements:
    pip install sounddevice  # For real-time audio playback
"""

import argparse
import sys
import torch
import numpy as np

# Optional: for real-time playback
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Note: sounddevice not installed. Install with 'pip install sounddevice' for real-time playback.")


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Example")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu/mps)")
    parser.add_argument("--text", type=str,
                        default="Hello! This is a demonstration of streaming text to speech. "
                                "Notice how the audio starts playing before the entire sentence is generated.",
                        help="Text to synthesize")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to reference audio file (6-15 seconds). Uses default voice if not provided.")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Tokens per audio chunk (default: 50, ~2 seconds)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to WAV file (optional)")
    parser.add_argument("--turbo", action="store_true",
                        help="Use Turbo model (faster, recommended)")
    parser.add_argument("--no-play", action="store_true",
                        help="Don't play audio (just print metrics)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model on {args.device}...")
    if args.turbo:
        from chatterbox import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(args.device)
    else:
        from chatterbox import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(args.device)

    print(f"Generating speech for: \"{args.text[:50]}{'...' if len(args.text) > 50 else ''}\"")
    print(f"Chunk size: {args.chunk_size} tokens (~{args.chunk_size / 25:.1f}s per chunk)")
    print("-" * 60)

    # Collect all audio for optional saving
    all_audio = []

    # Stream generation
    stream_kwargs = {
        "text": args.text,
        "chunk_size": args.chunk_size,
        "temperature": args.temperature,
        "print_metrics": True,
    }
    if args.reference:
        stream_kwargs["audio_prompt_path"] = args.reference

    for audio_chunk, metrics in model.generate_stream(**stream_kwargs):
        audio_np = audio_chunk.cpu().numpy()
        all_audio.append(audio_np)

        # Play audio chunk in real-time
        if HAS_SOUNDDEVICE and not args.no_play:
            sd.play(audio_np, samplerate=model.sr)
            sd.wait()

    print("-" * 60)
    print("Generation complete!")

    # Save to file if requested
    if args.output:
        import scipy.io.wavfile as wav
        full_audio = np.concatenate(all_audio)
        wav.write(args.output, model.sr, (full_audio * 32767).astype(np.int16))
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
