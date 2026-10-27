#!/usr/bin/env python3
"""Readers and visualization helpers for PathLearn modality data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


@dataclass
class GazeSample:
    timestamp: datetime | None
    image_x: float
    image_y: float
    scale: float | None


class GazeReader:
    """Reader for eye-tracking JSONL split files (`level == eye_movement`)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = self._load()

    def _load(self) -> list[GazeSample]:
        samples: list[GazeSample] = []
        for row in _read_jsonl(self.path):
            points = row.get("message")
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                image = point.get("relativeImage")
                if not isinstance(image, dict):
                    continue
                x = image.get("x")
                y = image.get("y")
                if x is None or y is None:
                    continue
                samples.append(
                    GazeSample(
                        timestamp=_parse_iso8601(point.get("timestamp")),
                        image_x=float(x),
                        image_y=float(y),
                        scale=float(point["scale"]) if point.get("scale") is not None else None,
                    )
                )
        return samples

    def visualize(self, *, max_points: int = 5000, figsize: tuple[int, int] = (8, 6)):
        """Scatter gaze points in image space (colored by time index)."""
        if not self.samples:
            raise ValueError(f"No gaze samples found in {self.path}")

        points = self.samples[-max_points:]
        xs = [p.image_x for p in points]
        ys = [p.image_y for p in points]
        colors = list(range(len(points)))

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(xs, ys, c=colors, s=7, cmap="viridis", alpha=0.7)
        ax.invert_yaxis()
        ax.set_title(f"Gaze Points ({len(points)} samples)")
        ax.set_xlabel("Image X")
        ax.set_ylabel("Image Y")
        fig.colorbar(sc, ax=ax, label="Time Index")
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def load_eye_points_from_segment_raw(
        segment_raw_path: str | Path,
    ) -> tuple[dict[str, float], list[tuple[float, float]]]:
        from final_outputs.gaze_attention import load_eye_points_from_segment_raw as _load_raw

        return _load_raw(segment_raw_path)

    @classmethod
    def attention_tier_presets(cls) -> dict[str, dict[str, float | str]]:
        from final_outputs.gaze_attention import attention_tier_presets as _presets

        return _presets()

    def visualize_attention_contour(
        self,
        *,
        points: list[tuple[float, float]] | None = None,
        bbox: dict[str, float] | None = None,
        max_points: int = 5000,
        grid_size: int = 256,
        attention_tier: str | None = None,
        gaussian_sigma: float | None = None,
        threshold_quantile: float | None = None,
        show_points: bool = True,
        figsize: tuple[int, int] = (8, 6),
    ):
        from final_outputs.gaze_attention import render_attention_contour as _render_attention_contour

        if points is None:
            if not self.samples:
                raise ValueError(f"No gaze samples found in {self.path}")
            selected = self.samples[-max_points:]
            points = [(sample.image_x, sample.image_y) for sample in selected]
        else:
            points = points[-max_points:]
        if not points:
            raise ValueError("No gaze points provided.")

        if bbox is None:
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            pad_x = max(1.0, (max(xs) - min(xs)) * 0.05)
            pad_y = max(1.0, (max(ys) - min(ys)) * 0.05)
            x1, x2 = min(xs) - pad_x, max(xs) + pad_x
            y1, y2 = min(ys) - pad_y, max(ys) + pad_y
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        return _render_attention_contour(
            points=points,
            bbox=bbox,
            max_points=max_points,
            grid_size=grid_size,
            attention_tier=attention_tier,
            gaussian_sigma=gaussian_sigma,
            threshold_quantile=threshold_quantile,
            show_points=show_points,
            figsize=figsize,
        )

    @classmethod
    def visualize_attention_contour_from_segment_raw(
        cls,
        segment_raw_path: str | Path,
        *,
        max_points: int = 5000,
        grid_size: int = 256,
        attention_tier: str | None = None,
        gaussian_sigma: float | None = None,
        threshold_quantile: float | None = None,
        show_points: bool = True,
        figsize: tuple[int, int] = (8, 6),
    ):
        from final_outputs.gaze_attention import (
            render_attention_contour_from_segment_raw as _render_attention_contour_from_segment_raw,
        )

        return _render_attention_contour_from_segment_raw(
            segment_raw_path,
            max_points=max_points,
            grid_size=grid_size,
            attention_tier=attention_tier,
            gaussian_sigma=gaussian_sigma,
            threshold_quantile=threshold_quantile,
            show_points=show_points,
            figsize=figsize,
        )


@dataclass
class MouseSample:
    timestamp: datetime | None
    image_x: float
    image_y: float
    magnification: float | None


class MouseReader:
    """Reader for mouse JSONL split files (`level == mouse`)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = self._load()

    def _load(self) -> list[MouseSample]:
        samples: list[MouseSample] = []
        for row in _read_jsonl(self.path):
            message = row.get("message") or {}
            coords = message.get("coordinates") if isinstance(message, dict) else None
            image = coords.get("image") if isinstance(coords, dict) else None
            if not isinstance(image, dict):
                continue
            x = image.get("x")
            y = image.get("y")
            if x is None or y is None:
                continue
            ts = message.get("timestamp") if isinstance(message, dict) else None
            magnification = message.get("magnification") if isinstance(message, dict) else None
            samples.append(
                MouseSample(
                    timestamp=_parse_iso8601(ts if isinstance(ts, str) else row.get("timestamp")),
                    image_x=float(x),
                    image_y=float(y),
                    magnification=float(magnification) if magnification is not None else None,
                )
            )
        return samples

    def visualize(self, *, max_points: int = 5000, figsize: tuple[int, int] = (8, 6)):
        """Plot mouse path in image space with start/end markers."""
        if not self.samples:
            raise ValueError(f"No mouse samples found in {self.path}")

        points = self.samples[-max_points:]
        xs = [p.image_x for p in points]
        ys = [p.image_y for p in points]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xs, ys, lw=1.0, alpha=0.7, color="#1f77b4")
        ax.scatter(xs[0], ys[0], color="green", s=35, label="start")
        ax.scatter(xs[-1], ys[-1], color="red", s=35, label="end")
        ax.invert_yaxis()
        ax.set_title(f"Mouse Path ({len(points)} samples)")
        ax.set_xlabel("Image X")
        ax.set_ylabel("Image Y")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax


@dataclass
class ViewportSample:
    timestamp: datetime | None
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class ViewportReader:
    """Reader for viewport JSONL split files (`level == coordinates`)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = self._load()

    def _load(self) -> list[ViewportSample]:
        samples: list[ViewportSample] = []
        for row in _read_jsonl(self.path):
            message = row.get("message") or {}
            coordinates = message.get("coordinates") if isinstance(message, dict) else None
            image = coordinates.get("image") if isinstance(coordinates, dict) else None
            if not isinstance(image, dict):
                continue
            x1, y1, x2, y2 = image.get("x1"), image.get("y1"), image.get("x2"), image.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            ts = message.get("timestamp") if isinstance(message, dict) else None
            samples.append(
                ViewportSample(
                    timestamp=_parse_iso8601(ts if isinstance(ts, str) else row.get("timestamp")),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return samples

    def visualize(
        self,
        *,
        max_samples: int = 300,
        rectangle_stride: int = 20,
        figsize: tuple[int, int] = (9, 7),
    ):
        """Plot viewport center path and sampled viewport rectangles."""
        if not self.samples:
            raise ValueError(f"No viewport samples found in {self.path}")

        samples = self.samples[-max_samples:]
        centers_x = [s.center_x for s in samples]
        centers_y = [s.center_y for s in samples]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(centers_x, centers_y, color="#d62728", lw=1.2, alpha=0.8, label="viewport center path")
        ax.scatter(centers_x[0], centers_y[0], color="green", s=45, label="start")
        ax.scatter(centers_x[-1], centers_y[-1], color="red", s=45, label="end")

        for idx in range(0, len(samples), max(1, rectangle_stride)):
            s = samples[idx]
            rect = Rectangle(
                (s.x1, s.y1),
                s.width,
                s.height,
                fill=False,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.3,
            )
            ax.add_patch(rect)

        ax.invert_yaxis()
        ax.set_title(f"Viewport Trajectory ({len(samples)} samples)")
        ax.set_xlabel("Image X")
        ax.set_ylabel("Image Y")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig, ax


@dataclass
class VoiceSegment:
    start_sec: float | None
    end_sec: float | None
    text: str


class VoiceReader:
    """Reader for transcript JSON or JSONL files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.segments = self._load()

    def _load(self) -> list[VoiceSegment]:
        if self.path.suffix.lower() == ".jsonl":
            rows = _read_jsonl(self.path)
        else:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict):
                if isinstance(raw.get("segments"), list):
                    rows = raw["segments"]
                elif isinstance(raw.get("results"), list):
                    rows = raw["results"]
                else:
                    rows = [raw]
            else:
                rows = []

        segments: list[VoiceSegment] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            start_sec = row.get("start") if row.get("start") is not None else row.get("start_sec")
            end_sec = row.get("end") if row.get("end") is not None else row.get("end_sec")
            segments.append(
                VoiceSegment(
                    start_sec=float(start_sec) if start_sec is not None else None,
                    end_sec=float(end_sec) if end_sec is not None else None,
                    text=text,
                )
            )
        return segments

    def iter_text(self) -> Iterable[str]:
        for segment in self.segments:
            yield segment.text
