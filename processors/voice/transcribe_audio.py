#!/usr/bin/env python3
"""CLI for backend-agnostic voice transcription."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from processors.voice.VoiceData import VoiceData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio using either OpenAI API or local Whisper and write "
            "a unified PathLearn transcript JSON."
        )
    )
    parser.add_argument("--input", required=True, help="Input audio path (.webm, .mp3, etc.)")
    parser.add_argument("--output", required=True, help="Output transcript JSON path")
    parser.add_argument(
        "--backend",
        choices=["openai_api", "local_whisper"],
        default="local_whisper",
        help="Transcription backend",
    )
    parser.add_argument("--model", default=None, help="Backend model name")
    parser.add_argument("--language", default=None, help="Optional language hint (e.g., en)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key override")
    parser.add_argument("--device", default="auto", help="Local Whisper device (auto/cpu/cuda)")
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Local Whisper compute type (default/int8/float16/etc.)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voice = VoiceData(args.input)

    if args.backend == "openai_api":
        transcript = voice.transcribe_openai_api(
            model=args.model or "gpt-4o-mini-transcribe",
            api_key=args.api_key,
            language=args.language,
        )
    else:
        transcript = voice.transcribe_local_whisper(
            model=args.model or "large-v3",
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
        )

    output_path = transcript.write_json(args.output)
    print(f"backend={transcript.backend}")
    print(f"segments={len(transcript.segments)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
