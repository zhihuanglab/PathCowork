#!/usr/bin/env python3
"""Gaze data reader class for PathLearn gaze JSONL files."""

from __future__ import annotations

import json
import math
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from final_outputs.gaze_attention import save_attention_contour_from_segment_raw
from processors.gaze.attention_polygon import (
    build_segment_gaze_multipolygon_artifact,
    validate_gaze_multipolygon_artifact,
    write_gaze_multipolygon_artifact,
)
from processors.modality_readers import GazeReader


@dataclass
class FixationEvent:
    """One fixation interval detected by the hybrid IVT-IDT filter."""

    start_index: int
    end_index: int
    start_time_sec: float
    end_time_sec: float
    duration_ms: float
    sample_count: int
    centroid_x: float
    centroid_y: float
    dispersion: float


class GazeData(GazeReader):
    """Reader + IVT-IDT fixation filtering helpers for PathLearn gaze data."""

    def __init__(self, path: str | Path):
        super().__init__(path)

    @staticmethod
    def _crop_union_canvas(
        image: Image.Image,
        canvas_crop_insets: tuple[int, int, int, int] | None,
    ) -> Image.Image:
        if canvas_crop_insets is None:
            return image
        top, bottom, left, right = canvas_crop_insets
        width, height = image.size
        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = max(x1 + 1, width - max(0, int(right)))
        y2 = max(y1 + 1, height - max(0, int(bottom)))
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _to_epoch_seconds(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _build_time_axis(
        num_points: int,
        *,
        timestamps: list[Any] | None,
        sample_rate_hz: float,
    ) -> list[float]:
        if num_points <= 0:
            return []
        fallback_dt = 1.0 / max(float(sample_rate_hz), 1e-6)
        if not timestamps or len(timestamps) != num_points:
            return [i * fallback_dt for i in range(num_points)]

        parsed = [GazeData._to_epoch_seconds(ts) for ts in timestamps]
        if any(ts is None for ts in parsed):
            return [i * fallback_dt for i in range(num_points)]

        axis: list[float] = [float(parsed[0])]  # type: ignore[arg-type]
        for i in range(1, num_points):
            cur = float(parsed[i])  # type: ignore[arg-type]
            prev = axis[-1]
            if cur <= prev:
                cur = prev + fallback_dt
            axis.append(cur)
        return axis

    @staticmethod
    def _dispersion(points: list[tuple[float, float]], start: int, end: int) -> float:
        xs = [points[i][0] for i in range(start, end + 1)]
        ys = [points[i][1] for i in range(start, end + 1)]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    @staticmethod
    def _find_low_velocity_runs(low_velocity_mask: list[bool]) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for idx, flag in enumerate(low_velocity_mask):
            if flag and run_start is None:
                run_start = idx
                continue
            if not flag and run_start is not None:
                runs.append((run_start, idx - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(low_velocity_mask) - 1))
        return runs

    @classmethod
    def ivt_idt_hybrid_filter(
        cls,
        points: list[tuple[float, float]],
        *,
        timestamps: list[Any] | None = None,
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
    ) -> dict[str, Any]:
        """
        Hybrid fixation filtering:
        1) IVT gate with velocity threshold.
        2) IDT refinement inside low-velocity windows.
        """
        if not points:
            return {
                "raw_point_count": 0,
                "processed_point_count": 0,
                "retention_ratio": 0.0,
                "velocity_threshold_px_per_sec": velocity_threshold_px_per_sec,
                "dispersion_threshold_px": dispersion_threshold_px,
                "min_duration_ms": min_duration_ms,
                "sample_rate_hz": sample_rate_hz,
                "fixation_count": 0,
                "velocities_px_per_sec": [],
                "kept_indices": [],
                "filtered_points": [],
                "fixations": [],
            }

        n = len(points)
        times = cls._build_time_axis(
            n,
            timestamps=timestamps,
            sample_rate_hz=sample_rate_hz,
        )
        fallback_dt = 1.0 / max(float(sample_rate_hz), 1e-6)

        velocities = [0.0] * n
        for i in range(1, n):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            dt = times[i] - times[i - 1]
            if dt <= 0:
                dt = fallback_dt
            velocities[i] = math.hypot(dx, dy) / dt

        low_velocity = [v <= float(velocity_threshold_px_per_sec) for v in velocities]
        runs = cls._find_low_velocity_runs(low_velocity)
        min_duration_sec = float(min_duration_ms) / 1000.0

        fixation_events: list[FixationEvent] = []
        kept_indices: set[int] = set()

        for run_start, run_end in runs:
            cursor = run_start
            while cursor <= run_end:
                min_end = cursor
                while min_end <= run_end and (times[min_end] - times[cursor]) < min_duration_sec:
                    min_end += 1
                if min_end > run_end:
                    break

                if cls._dispersion(points, cursor, min_end) > float(dispersion_threshold_px):
                    cursor += 1
                    continue

                best_end = min_end
                expand = min_end + 1
                while expand <= run_end:
                    if cls._dispersion(points, cursor, expand) > float(dispersion_threshold_px):
                        break
                    best_end = expand
                    expand += 1

                fix_points = points[cursor : best_end + 1]
                xs = [p[0] for p in fix_points]
                ys = [p[1] for p in fix_points]
                fixation = FixationEvent(
                    start_index=cursor,
                    end_index=best_end,
                    start_time_sec=times[cursor],
                    end_time_sec=times[best_end],
                    duration_ms=(times[best_end] - times[cursor]) * 1000.0,
                    sample_count=len(fix_points),
                    centroid_x=sum(xs) / len(xs),
                    centroid_y=sum(ys) / len(ys),
                    dispersion=cls._dispersion(points, cursor, best_end),
                )
                fixation_events.append(fixation)
                for idx in range(cursor, best_end + 1):
                    kept_indices.add(idx)
                cursor = best_end + 1

        kept_sorted = sorted(kept_indices)
        filtered_points = [points[i] for i in kept_sorted]
        raw_count = len(points)
        processed_count = len(filtered_points)
        retention = processed_count / raw_count if raw_count else 0.0

        return {
            "raw_point_count": raw_count,
            "processed_point_count": processed_count,
            "retention_ratio": retention,
            "velocity_threshold_px_per_sec": velocity_threshold_px_per_sec,
            "dispersion_threshold_px": dispersion_threshold_px,
            "min_duration_ms": min_duration_ms,
            "sample_rate_hz": sample_rate_hz,
            "fixation_count": len(fixation_events),
            "velocities_px_per_sec": velocities,
            "kept_indices": kept_sorted,
            "filtered_points": filtered_points,
            "fixations": [asdict(fix) for fix in fixation_events],
        }

    def filter_loaded_samples(
        self,
        *,
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
    ) -> dict[str, Any]:
        points = [(sample.image_x, sample.image_y) for sample in self.samples]
        timestamps = [sample.timestamp for sample in self.samples]
        return self.ivt_idt_hybrid_filter(
            points,
            timestamps=timestamps,
            velocity_threshold_px_per_sec=velocity_threshold_px_per_sec,
            dispersion_threshold_px=dispersion_threshold_px,
            min_duration_ms=min_duration_ms,
            sample_rate_hz=sample_rate_hz,
        )

    @staticmethod
    def _draw_gaze_points_overlay(
        *,
        union_image_path: Path,
        bbox: dict[str, float],
        points: list[tuple[float, float]],
        output_path: Path,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        point_radius: int = 3,
        point_color: tuple[int, int, int, int] = (32, 201, 151, 130),
    ) -> Path:
        image = Image.open(union_image_path).convert("RGBA")
        image = GazeData._crop_union_canvas(image, canvas_crop_insets)
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        bw = max(x2 - x1, 1e-6)
        bh = max(y2 - y1, 1e-6)

        for x, y in points:
            if x < x1 or x > x2 or y < y1 or y > y2:
                continue
            px = (x - x1) / bw * width
            py = (y - y1) / bh * height
            draw.ellipse(
                (px - point_radius, py - point_radius, px + point_radius, py + point_radius),
                fill=point_color,
                outline=None,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path)
        return output_path

    @classmethod
    def _save_boundary_from_points(
        cls,
        *,
        points: list[tuple[float, float]],
        bbox: dict[str, float],
        union_image_path: Path,
        output_path: Path,
        attention_tier: str = "mid",
        grid_size: int = 256,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        boundary_color: str = "#d62828",
        boundary_width: int = 3,
    ) -> Path:
        if not points:
            # No eye points for this segment: emit cropped union image as boundary placeholder.
            image = Image.open(union_image_path).convert("RGB")
            image = GazeData._crop_union_canvas(image, canvas_crop_insets)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)
            return out

        payload = {
            "boxes_final": {
                "x1": float(bbox["x1"]),
                "y1": float(bbox["y1"]),
                "x2": float(bbox["x2"]),
                "y2": float(bbox["y2"]),
            },
            "eye": [{"x": float(x), "y": float(y)} for x, y in points],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=True) as handle:
            json.dump(payload, handle, ensure_ascii=True)
            handle.flush()
            cropped = Image.open(union_image_path).convert("RGB")
            cropped = GazeData._crop_union_canvas(cropped, canvas_crop_insets)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as img_handle:
                cropped.save(img_handle.name)
                save_attention_contour_from_segment_raw(
                    handle.name,
                    output_path,
                    union_image_path=img_handle.name,
                    render_mode="boundary_overlay",
                    grid_size=grid_size,
                    attention_tier=attention_tier,
                    boundary_color=boundary_color,
                    boundary_width=boundary_width,
                )
        return output_path

    @classmethod
    def export_segment_raw_ivt_idt_comparison(
        cls,
        segment_raw_path: str | Path,
        *,
        output_dir: str | Path,
        union_image_path: str | Path | None = None,
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
        attention_tier: str = "mid",
        grid_size: int = 256,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        point_radius: int = 3,
        boundary_color: str = "#d62828",
        boundary_width: int = 3,
    ) -> dict[str, Any]:
        segment_raw_path = Path(segment_raw_path)
        if union_image_path is None:
            union_image_path = segment_raw_path.with_name(
                segment_raw_path.name.replace("_raw.json", "_union.png")
            )
        union_image_path = Path(union_image_path)
        if not union_image_path.exists():
            raise FileNotFoundError(f"Union image not found: {union_image_path}")

        bbox, points = cls.load_eye_points_from_segment_raw(segment_raw_path)
        filtered = cls.ivt_idt_hybrid_filter(
            points,
            timestamps=None,
            velocity_threshold_px_per_sec=velocity_threshold_px_per_sec,
            dispersion_threshold_px=dispersion_threshold_px,
            min_duration_ms=min_duration_ms,
            sample_rate_hz=sample_rate_hz,
        )
        processed_points = filtered["filtered_points"]
        fallback_to_raw = False
        if not processed_points:
            # Keep segment in downstream panels even if thresholds are too strict.
            processed_points = points
            fallback_to_raw = True

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = segment_raw_path.stem.replace("_raw", "")

        initial_gaze_path = output_dir / f"{stem}_initial_gaze.png"
        processed_gaze_path = output_dir / f"{stem}_processed_gaze.png"
        initial_boundary_path = output_dir / f"{stem}_initial_gaze_attention_boundary.png"
        processed_boundary_path = output_dir / f"{stem}_processed_gaze_attention_boundary.png"

        cls._draw_gaze_points_overlay(
            union_image_path=union_image_path,
            bbox=bbox,
            points=points,
            output_path=initial_gaze_path,
            canvas_crop_insets=canvas_crop_insets,
            point_radius=point_radius,
        )
        cls._draw_gaze_points_overlay(
            union_image_path=union_image_path,
            bbox=bbox,
            points=processed_points,
            output_path=processed_gaze_path,
            canvas_crop_insets=canvas_crop_insets,
            point_radius=point_radius,
        )
        cls._save_boundary_from_points(
            points=points,
            bbox=bbox,
            union_image_path=union_image_path,
            output_path=initial_boundary_path,
            attention_tier=attention_tier,
            grid_size=grid_size,
            canvas_crop_insets=canvas_crop_insets,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )
        cls._save_boundary_from_points(
            points=processed_points,
            bbox=bbox,
            union_image_path=union_image_path,
            output_path=processed_boundary_path,
            attention_tier=attention_tier,
            grid_size=grid_size,
            canvas_crop_insets=canvas_crop_insets,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )

        filtered["segment_raw_path"] = str(segment_raw_path)
        filtered["union_image_path"] = str(union_image_path)
        filtered["bbox"] = bbox
        filtered["fallback_to_raw"] = fallback_to_raw
        filtered["output_paths"] = {
            "initial_gaze": str(initial_gaze_path),
            "processed_gaze": str(processed_gaze_path),
            "initial_boundary": str(initial_boundary_path),
            "processed_boundary": str(processed_boundary_path),
        }
        return filtered

    @classmethod
    def export_segment_raw_ivt_idt_4subplot(
        cls,
        segment_raw_path: str | Path,
        *,
        output_dir: str | Path,
        union_image_path: str | Path | None = None,
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
        attention_tier: str = "mid",
        grid_size: int = 256,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        point_radius: int = 3,
        boundary_color: str = "#d62828",
        boundary_width: int = 3,
    ) -> dict[str, Any]:
        """Export one 2x2 plot for a segment: raw/processed gaze and boundaries."""
        result = cls.export_segment_raw_ivt_idt_comparison(
            segment_raw_path,
            output_dir=output_dir,
            union_image_path=union_image_path,
            velocity_threshold_px_per_sec=velocity_threshold_px_per_sec,
            dispersion_threshold_px=dispersion_threshold_px,
            min_duration_ms=min_duration_ms,
            sample_rate_hz=sample_rate_hz,
            attention_tier=attention_tier,
            grid_size=grid_size,
            canvas_crop_insets=canvas_crop_insets,
            point_radius=point_radius,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )

        segment_raw_path = Path(segment_raw_path)
        stem = segment_raw_path.stem.replace("_raw", "")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        panel_path = output_dir / f"{stem}_ivt_idt_4subplot.png"

        layout = [
            ("Initial Gaze", result["output_paths"]["initial_gaze"]),
            ("Initial Gaze Attention Boundary", result["output_paths"]["initial_boundary"]),
            ("Processed Gaze", result["output_paths"]["processed_gaze"]),
            ("Processed Gaze Attention Boundary", result["output_paths"]["processed_boundary"]),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (title, img_path) in zip(axes.flatten(), layout):
            ax.imshow(mpimg.imread(img_path))
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        fig.suptitle(
            (
                f"{stem} | points {result['raw_point_count']} -> {result['processed_point_count']} "
                f"| fixations={result['fixation_count']} "
                f"| keep={result['retention_ratio']:.3f}"
            ),
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(panel_path, dpi=160)
        plt.close(fig)

        result["output_paths"]["comparison_4subplot"] = str(panel_path)
        return result

    @classmethod
    def export_dataset_ivt_idt_10case_panel(
        cls,
        dataset_root: str | Path,
        *,
        output_dir: str | Path,
        max_cases: int = 10,
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
        attention_tier: str = "mid",
        grid_size: int = 256,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        point_radius: int = 3,
        boundary_color: str = "#d62828",
        boundary_width: int = 3,
    ) -> dict[str, Any]:
        """
        Batch-export 4-subplot comparisons and one single rowsx4 panel.
        Each row = one segment, columns = initial gaze / initial boundary / processed gaze / processed boundary.
        """
        dataset_root = Path(dataset_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_paths = sorted(dataset_root.glob("seg_*_raw.json"))
        if not raw_paths:
            raise FileNotFoundError(f"No seg_*_raw.json found in {dataset_root}")
        selected = raw_paths[: max(1, max_cases)]

        cases: list[dict[str, Any]] = []
        for raw_path in selected:
            case = cls.export_segment_raw_ivt_idt_4subplot(
                raw_path,
                output_dir=output_dir,
                velocity_threshold_px_per_sec=velocity_threshold_px_per_sec,
                dispersion_threshold_px=dispersion_threshold_px,
                min_duration_ms=min_duration_ms,
                sample_rate_hz=sample_rate_hz,
                attention_tier=attention_tier,
                grid_size=grid_size,
                canvas_crop_insets=canvas_crop_insets,
                point_radius=point_radius,
                boundary_color=boundary_color,
                boundary_width=boundary_width,
            )
            cases.append(case)

        n = len(cases)
        headers = [
            "Initial Gaze",
            "Initial Boundary",
            "Processed Gaze",
            "Processed Boundary",
        ]
        fig, axes = plt.subplots(n, 4, figsize=(18, max(3 * n, 8)))
        if n == 1:
            axes = [axes]  # type: ignore[assignment]

        for row_idx, case in enumerate(cases):
            row_axes = axes[row_idx]
            imgs = [
                case["output_paths"]["initial_gaze"],
                case["output_paths"]["initial_boundary"],
                case["output_paths"]["processed_gaze"],
                case["output_paths"]["processed_boundary"],
            ]
            seg_name = Path(case["segment_raw_path"]).stem.replace("_raw", "")
            for col_idx, (ax, img_path) in enumerate(zip(row_axes, imgs)):
                ax.imshow(mpimg.imread(img_path))
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(headers[col_idx], fontsize=11)
            row_axes[0].set_ylabel(
                (
                    f"{seg_name}\n"
                    f"{case['raw_point_count']}->{case['processed_point_count']} "
                    f"fix={case['fixation_count']}"
                ),
                rotation=0,
                ha="right",
                va="center",
                labelpad=35,
                fontsize=9,
            )

        fig.suptitle(
            (
                f"IVT-IDT Hybrid Comparison ({n} cases) | "
                f"v<={velocity_threshold_px_per_sec:.0f}px/s, "
                f"disp<={dispersion_threshold_px:.0f}px, "
                f"min_dur={min_duration_ms:.0f}ms, tier={attention_tier}"
            ),
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        big_panel_path = output_dir / f"ivt_idt_{n}cases_rowsx4.png"
        fig.savefig(big_panel_path, dpi=170)
        plt.close(fig)

        return {
            "case_count": n,
            "panel_path": str(big_panel_path),
            "cases": [
                {
                    "segment_raw_path": case["segment_raw_path"],
                    "raw_point_count": case["raw_point_count"],
                    "processed_point_count": case["processed_point_count"],
                    "fixation_count": case["fixation_count"],
                    "retention_ratio": case["retention_ratio"],
                    "comparison_4subplot": case["output_paths"]["comparison_4subplot"],
                }
                for case in cases
            ],
        }

    @classmethod
    def export_dataset_ivt_idt_numbered_panel(
        cls,
        dataset_root: str | Path,
        *,
        output_dir: str | Path,
        max_cases: int = 0,
        boundary_source: str = "processed",
        refined_text_jsonl: str | Path | None = None,
        text_preference: tuple[str, ...] = ("text_refined", "text", "text_raw"),
        velocity_threshold_px_per_sec: float = 1800.0,
        dispersion_threshold_px: float = 220.0,
        min_duration_ms: float = 90.0,
        sample_rate_hz: float = 30.0,
        attention_tier: str = "mid",
        grid_size: int = 256,
        canvas_crop_insets: tuple[int, int, int, int] | None = None,
        point_radius: int = 3,
        boundary_color: str = "#d62828",
        boundary_width: int = 3,
        cell_width: int = 300,
    ) -> dict[str, Any]:
        """
        Export one numbered manuscript panel:
        - Rows: segments (default all)
        - Col1: initial gaze
        - Col2: processed gaze
        - Col3: boundary (processed or initial)
        """
        boundary_source = boundary_source.strip().lower()
        if boundary_source not in {"processed", "initial"}:
            raise ValueError("boundary_source must be 'processed' or 'initial'")

        dataset_root = Path(dataset_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_paths = sorted(dataset_root.glob("seg_*_raw.json"))
        if not raw_paths:
            raise FileNotFoundError(f"No seg_*_raw.json found in {dataset_root}")
        if max_cases > 0:
            raw_paths = raw_paths[:max_cases]

        text_lookup: dict[str, str] = {}
        if refined_text_jsonl is not None:
            path = Path(refined_text_jsonl)
            if not path.exists():
                raise FileNotFoundError(f"Refined text JSONL not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = str(row.get("segment_key", "")).strip()
                    if not key:
                        continue
                    chosen = ""
                    for field in text_preference:
                        value = row.get(field)
                        if isinstance(value, str) and value.strip():
                            chosen = value.strip()
                            break
                    if chosen:
                        text_lookup[key] = chosen

        cases: list[dict[str, Any]] = []
        for row_idx, raw_path in enumerate(raw_paths, start=1):
            case = cls.export_segment_raw_ivt_idt_comparison(
                raw_path,
                output_dir=output_dir,
                velocity_threshold_px_per_sec=velocity_threshold_px_per_sec,
                dispersion_threshold_px=dispersion_threshold_px,
                min_duration_ms=min_duration_ms,
                sample_rate_hz=sample_rate_hz,
                attention_tier=attention_tier,
                grid_size=grid_size,
                canvas_crop_insets=canvas_crop_insets,
                point_radius=point_radius,
                boundary_color=boundary_color,
                boundary_width=boundary_width,
            )
            seg_key = Path(case["segment_raw_path"]).stem.replace("_raw", "")
            case["row_number"] = row_idx
            case["segment_key"] = seg_key
            case["left_text"] = text_lookup.get(seg_key.replace("seg_", ""), "")
            cases.append(case)

        ref_image = Image.open(cases[0]["output_paths"]["initial_gaze"]).convert("RGB")
        aspect = ref_image.height / max(ref_image.width, 1)
        cell_height = max(1, int(round(cell_width * aspect)))

        columns = ["Initial Gaze", "Processed Gaze", f"{boundary_source.title()} Boundary"]
        left_label_width = 520
        col_gap = 12
        row_gap = 10
        header_h = 52
        panel_w = left_label_width + (3 * cell_width) + (4 * col_gap)
        panel_h = header_h + ((cell_height + row_gap) * len(cases)) + row_gap

        canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for col_idx, title in enumerate(columns):
            x = left_label_width + col_gap + col_idx * (cell_width + col_gap)
            draw.text((x + 8, 18), title, fill=(15, 15, 15), font=font)

        resize_mode = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

        rows_meta: list[dict[str, Any]] = []
        for case in cases:
            row_number = int(case["row_number"])
            y = header_h + row_gap + (row_number - 1) * (cell_height + row_gap)
            seg_key = str(case["segment_key"])
            label = f"{row_number:02d}"
            draw.text((10, y + 8), label, fill=(0, 0, 0), font=font)
            draw.text((10, y + 24), seg_key, fill=(70, 70, 70), font=font)
            left_text = str(case.get("left_text") or "").strip()
            if left_text:
                wrapped = textwrap.wrap(left_text, width=72)
                wrapped = wrapped[:8]
                text_y = y + 44
                for line in wrapped:
                    draw.text((10, text_y), line, fill=(15, 15, 15), font=font)
                    text_y += 12

            if boundary_source == "processed":
                boundary_path = case["output_paths"]["processed_boundary"]
            else:
                boundary_path = case["output_paths"]["initial_boundary"]

            row_images = [
                case["output_paths"]["initial_gaze"],
                case["output_paths"]["processed_gaze"],
                boundary_path,
            ]
            for col_idx, img_path in enumerate(row_images):
                x = left_label_width + col_gap + col_idx * (cell_width + col_gap)
                img = Image.open(img_path).convert("RGB").resize((cell_width, cell_height), resize_mode)
                canvas.paste(img, (x, y))
                draw.rectangle(
                    (x, y, x + cell_width - 1, y + cell_height - 1),
                    outline=(210, 210, 210),
                    width=1,
                )

            rows_meta.append(
                {
                    "row_number": row_number,
                    "segment_key": seg_key,
                    "segment_raw_path": case["segment_raw_path"],
                    "raw_point_count": case["raw_point_count"],
                    "processed_point_count": case["processed_point_count"],
                    "fixation_count": case["fixation_count"],
                    "retention_ratio": case["retention_ratio"],
                    "left_text": left_text,
                    "initial_gaze": case["output_paths"]["initial_gaze"],
                    "processed_gaze": case["output_paths"]["processed_gaze"],
                    "boundary": boundary_path,
                }
            )

        panel_path = output_dir / f"ivt_idt_numbered_{len(cases)}rows_3cols_{boundary_source}_boundary.png"
        canvas.save(panel_path)

        mapping = {
            "format": "pathlearn.ivt_idt.numbered_panel.v1",
            "case_count": len(cases),
            "boundary_source": boundary_source,
            "panel_path": str(panel_path),
            "rows": rows_meta,
            "refined_text_jsonl": str(refined_text_jsonl) if refined_text_jsonl is not None else None,
        }
        mapping_path = output_dir / f"ivt_idt_numbered_{len(cases)}rows_row_mapping.json"
        mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {
            "case_count": len(cases),
            "panel_path": str(panel_path),
            "mapping_path": str(mapping_path),
            "rows": rows_meta,
        }

    @classmethod
    def export_segment_raw_multipolygon_json(
        cls,
        segment_raw_path: str | Path,
        *,
        output_path: str | Path,
        image_file_name: str | None = None,
        case_id: str | None = None,
        slide_id: str | None = None,
        segment_key: str | None = None,
        max_points: int = 5000,
        grid_size: int = 256,
        attention_tier: str = "mid",
        gaussian_sigma: float | None = None,
        threshold_quantile: float | None = None,
        min_component_area_ratio: float = 0.003,
        max_components: int = 3,
    ) -> dict[str, Any]:
        """
        Export one segment's gaze artifact with compact multipolygon geometry.

        Output JSON contract fields:
        - image_file_name, case_id, slide_id, segment_key
        - roi_rect_source_wsi_px
        - points_source_wsi_px, points_roi_norm
        - multipolygon_source_wsi_px, multipolygon_roi_norm
        """
        segment_raw_path = Path(segment_raw_path)
        if image_file_name is None:
            image_file_name = segment_raw_path.with_name(
                segment_raw_path.name.replace("_raw.json", "_union.png")
            ).name

        if segment_key is None:
            name = segment_raw_path.stem
            segment_key = name.split("_")[1] if name.startswith("seg_") and "_" in name else None

        payload = build_segment_gaze_multipolygon_artifact(
            segment_raw_path,
            image_file_name=image_file_name,
            case_id=case_id,
            slide_id=slide_id,
            segment_key=segment_key,
            max_points=max_points,
            grid_size=grid_size,
            attention_tier=attention_tier,
            gaussian_sigma=gaussian_sigma,
            threshold_quantile=threshold_quantile,
            min_component_area_ratio=min_component_area_ratio,
            max_components=max_components,
        )
        validate_gaze_multipolygon_artifact(payload)
        out = write_gaze_multipolygon_artifact(payload, output_path)
        payload["output_path"] = str(out)
        return payload

    @classmethod
    def export_dataset_multipolygon_json(
        cls,
        dataset_root: str | Path,
        *,
        output_dir: str | Path,
        max_cases: int = 0,
        case_id: str | None = None,
        slide_id: str | None = None,
        attention_tier: str = "mid",
        grid_size: int = 256,
    ) -> dict[str, Any]:
        """Batch-export multipolygon JSON artifacts from seg_*_raw.json dataset."""
        dataset_root = Path(dataset_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_paths = sorted(dataset_root.glob("seg_*_raw.json"))
        if not raw_paths:
            raise FileNotFoundError(f"No seg_*_raw.json found in {dataset_root}")
        if max_cases > 0:
            raw_paths = raw_paths[:max_cases]

        outputs: list[str] = []
        for raw_path in raw_paths:
            seg_key = raw_path.stem.split("_")[1] if raw_path.stem.startswith("seg_") else None
            out_path = output_dir / f"{raw_path.stem.replace('_raw', '')}_gaze_multipolygon.json"
            payload = cls.export_segment_raw_multipolygon_json(
                raw_path,
                output_path=out_path,
                image_file_name=raw_path.with_name(raw_path.name.replace("_raw.json", "_union.png")).name,
                case_id=case_id,
                slide_id=slide_id,
                segment_key=seg_key,
                attention_tier=attention_tier,
                grid_size=grid_size,
            )
            outputs.append(str(payload["output_path"]))

        manifest = {
            "format": "pathlearn.gaze.multipolygon.dataset.v1",
            "count": len(outputs),
            "items": outputs,
        }
        manifest_path = output_dir / "gaze_multipolygon_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {
            "count": len(outputs),
            "manifest_path": str(manifest_path),
            "items": outputs,
        }
