#!/usr/bin/env python3
"""Voice transcription utilities with backend-agnostic output format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class VoiceTranscript:
    """Unified transcript container used across online and local backends."""

    task: str
    language: str | None
    duration: float | None
    text: str
    segments: list[dict[str, Any]]
    backend: str
    source_audio: str
    created_at: str
    output_format: str = "pathlearn.voice.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_format": self.output_format,
            "backend": self.backend,
            "source_audio": self.source_audio,
            "created_at": self.created_at,
            "task": self.task,
            "language": self.language,
            "duration": self.duration,
            "text": self.text,
            "segments": self.segments,
        }

    def write_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        return path


class VoiceData:
    """
    Voice transcription processor.

    This class does not invent a new ASR algorithm. It wraps existing backends
    (online API or local Whisper) and normalizes outputs to one format.
    """

    def __init__(self, audio_path: str | Path):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

    def transcribe_openai_api(
        self,
        *,
        model: str = "gpt-4o-mini-transcribe",
        api_key: str | None = None,
        language: str | None = None,
    ) -> VoiceTranscript:
        """
        Transcribe using OpenAI's speech-to-text API.

        Requires `openai` package and API key (via arg or OPENAI_API_KEY env var).
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI API transcription. "
                "Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=api_key)
        with self.audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                language=language,
            )

        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return self._normalize_payload(
            payload,
            backend=f"openai_api:{model}",
            source_audio=str(self.audio_path),
        )

    def transcribe_local_whisper(
        self,
        *,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
    ) -> VoiceTranscript:
        """
        Transcribe with local faster-whisper backend.

        Requires `faster-whisper` and local model weights.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for local transcription. "
                "Install with: pip install faster-whisper"
            ) from exc

        whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        segments_iter, info = whisper_model.transcribe(str(self.audio_path), language=language)

        segments: list[dict[str, Any]] = []
        full_text_parts: list[str] = []
        for idx, seg in enumerate(segments_iter):
            text = (seg.text or "").strip()
            full_text_parts.append(text)
            segments.append(
                {
                    "id": idx,
                    "text": text,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
                    "language": info.language,
                    "speaker": None,
                    "words": None,
                }
            )

        payload = {
            "task": "transcribe",
            "language": info.language,
            "duration": float(info.duration) if info.duration is not None else None,
            "text": " ".join(part for part in full_text_parts if part).strip(),
            "segments": segments,
        }
        return self._normalize_payload(
            payload,
            backend=f"local_whisper:{model}",
            source_audio=str(self.audio_path),
        )

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        *,
        backend: str,
        source_audio: str,
    ) -> VoiceTranscript:
        task = str(payload.get("task") or "transcribe")
        language = payload.get("language")
        duration = payload.get("duration")
        text = str(payload.get("text") or "").strip()

        raw_segments = payload.get("segments")
        segments: list[dict[str, Any]] = []
        if isinstance(raw_segments, list):
            for idx, seg in enumerate(raw_segments):
                if not isinstance(seg, dict):
                    continue
                segments.append(
                    {
                        "id": int(seg.get("id", idx)),
                        "text": str(seg.get("text", "")).strip(),
                        "start": self._float_or_none(seg.get("start")),
                        "end": self._float_or_none(seg.get("end")),
                        "avg_logprob": self._float_or_none(seg.get("avg_logprob")),
                        "language": seg.get("language", language),
                        "speaker": seg.get("speaker"),
                        "words": seg.get("words"),
                    }
                )

        if not text and segments:
            text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()

        return VoiceTranscript(
            task=task,
            language=str(language) if language is not None else None,
            duration=self._float_or_none(duration),
            text=text,
            segments=segments,
            backend=backend,
            source_audio=source_audio,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
