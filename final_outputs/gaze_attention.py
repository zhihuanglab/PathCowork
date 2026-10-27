#!/usr/bin/env python3
"""Case-level gaze attention contour generation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ATTENTION_TIER_PRESETS: dict[str, dict[str, float | str]] = {
    "high": {
        "gaussian_sigma": 10.0,
        "threshold_quantile": 0.90,
        "description": "smaller, stricter ROI",
    },
    "mid": {
        "gaussian_sigma": 16.0,
        "threshold_quantile": 0.82,
        "description": "balanced ROI size",
    },
    "low": {
        "gaussian_sigma": 24.0,
        "threshold_quantile": 0.72,
        "description": "larger, broader ROI",
    },
}


def attention_tier_presets() -> dict[str, dict[str, float | str]]:
    return {key: dict(value) for key, value in ATTENTION_TIER_PRESETS.items()}


def resolve_attention_params(
    *,
    attention_tier: str | None,
    gaussian_sigma: float | None,
    threshold_quantile: float | None,
) -> tuple[float, float, str]:
    if attention_tier is None:
        sigma = 4.0 if gaussian_sigma is None else float(gaussian_sigma)
        quantile = 0.85 if threshold_quantile is None else float(threshold_quantile)
        return sigma, quantile, "custom"

    key = attention_tier.strip().lower()
    if key not in ATTENTION_TIER_PRESETS:
        choices = ", ".join(sorted(ATTENTION_TIER_PRESETS))
        raise ValueError(f"Unknown attention_tier='{attention_tier}'. Choose from: {choices}")

    preset = ATTENTION_TIER_PRESETS[key]
    sigma = float(preset["gaussian_sigma"]) if gaussian_sigma is None else float(gaussian_sigma)
    quantile = float(preset["threshold_quantile"]) if threshold_quantile is None else float(threshold_quantile)
    return sigma, quantile, key


def load_eye_points_from_segment_raw(
    segment_raw_path: str | Path,
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    path = Path(segment_raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    bbox = payload.get("boxes_final")
    if not isinstance(bbox, dict):
        raise ValueError(f"Missing boxes_final in {path}")
    required = ("x1", "y1", "x2", "y2")
    if any(bbox.get(key) is None for key in required):
        raise ValueError(f"boxes_final must contain {required} in {path}")

    points: list[tuple[float, float]] = []
    for point in payload.get("eye") or []:
        if not isinstance(point, dict):
            continue
        x = point.get("x")
        y = point.get("y")
        if x is None or y is None:
            continue
        points.append((float(x), float(y)))

    return (
        {
            "x1": float(bbox["x1"]),
            "y1": float(bbox["y1"]),
            "x2": float(bbox["x2"]),
            "y2": float(bbox["y2"]),
        },
        points,
    )


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur2d(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.copy()
    kernel = _gaussian_kernel1d(sigma=sigma)
    temp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=image)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=temp)


def _compute_attention_components(
    *,
    points: list[tuple[float, float]],
    bbox: dict[str, float],
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = None,
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
    show_points: bool = True,
    figsize: tuple[int, int] = (8, 6),
):
    gaussian_sigma, threshold_quantile, resolved_tier = resolve_attention_params(
        attention_tier=attention_tier,
        gaussian_sigma=gaussian_sigma,
        threshold_quantile=threshold_quantile,
    )
    if grid_size < 16:
        raise ValueError("grid_size must be >= 16")
    if threshold_quantile <= 0 or threshold_quantile >= 1:
        raise ValueError("threshold_quantile must be between 0 and 1")
    if not points:
        raise ValueError("No gaze points provided.")

    points = points[-max_points:]
    try:
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bbox must contain numeric keys x1,y1,x2,y2") from exc

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox}")

    local_points: list[tuple[float, float]] = []
    for x, y in points:
        if x < x1 or x > x2 or y < y1 or y > y2:
            continue
        local_points.append((x - x1, y - y1))
    if not local_points:
        raise ValueError("No gaze points fall inside bbox.")

    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    xs_local = np.array([point[0] for point in local_points], dtype=float)
    ys_local = np.array([point[1] for point in local_points], dtype=float)
    gx = np.clip(np.rint(xs_local / width * (grid_size - 1)).astype(int), 0, grid_size - 1)
    gy = np.clip(np.rint(ys_local / height * (grid_size - 1)).astype(int), 0, grid_size - 1)
    np.add.at(heatmap, (gy, gx), 1.0)

    blurred = _gaussian_blur2d(heatmap, sigma=gaussian_sigma)
    if float(blurred.max()) <= 0:
        raise ValueError("Unable to build attention map from gaze points.")
    norm = blurred / float(blurred.max())
    nonzero = norm[norm > 0]
    threshold = float(np.quantile(nonzero, threshold_quantile)) if nonzero.size > 0 else 0.5
    threshold = min(max(threshold, 1e-6), 0.999)

    x_axis = np.linspace(0.0, width, grid_size)
    y_axis = np.linspace(0.0, height, grid_size)
    return {
        "width": width,
        "height": height,
        "norm": norm,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "threshold": threshold,
        "local_points": local_points,
        "resolved_tier": resolved_tier,
        "gaussian_sigma": gaussian_sigma,
        "threshold_quantile": threshold_quantile,
    }


def render_attention_contour(
    *,
    points: list[tuple[float, float]],
    bbox: dict[str, float],
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = None,
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
    show_points: bool = True,
    figsize: tuple[int, int] = (8, 6),
):
    comps = _compute_attention_components(
        points=points,
        bbox=bbox,
        max_points=max_points,
        grid_size=grid_size,
        attention_tier=attention_tier,
        gaussian_sigma=gaussian_sigma,
        threshold_quantile=threshold_quantile,
    )
    width = float(comps["width"])
    height = float(comps["height"])
    norm = comps["norm"]
    x_axis = comps["x_axis"]
    y_axis = comps["y_axis"]
    threshold = float(comps["threshold"])
    local_points = comps["local_points"]
    resolved_tier = str(comps["resolved_tier"])
    gaussian_sigma = float(comps["gaussian_sigma"])
    threshold_quantile = float(comps["threshold_quantile"])
    xs_local = np.array([point[0] for point in local_points], dtype=float)
    ys_local = np.array([point[1] for point in local_points], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        norm,
        extent=(0.0, width, height, 0.0),
        cmap="magma",
        alpha=0.28,
        interpolation="bilinear",
    )
    ax.contourf(
        x_axis,
        y_axis,
        norm,
        levels=[threshold, 1.0],
        colors=["#ff9f1c"],
        alpha=0.35,
        antialiased=True,
    )
    ax.contour(
        x_axis,
        y_axis,
        norm,
        levels=[threshold],
        colors=["#d62828"],
        linewidths=2.0,
    )
    if show_points:
        ax.scatter(
            xs_local,
            ys_local,
            s=8,
            color="#2ec4b6",
            alpha=0.5,
            edgecolors="none",
            label="gaze points",
        )
        ax.legend(loc="upper right")

    ax.set_title(
        (
            f"Gaze Attention Contour ({len(local_points)} points, grid={grid_size}, "
            f"tier={resolved_tier}, sigma={gaussian_sigma:.1f}, q={threshold_quantile:.2f})"
        )
    )
    ax.set_xlabel("ROI Local X")
    ax.set_ylabel("ROI Local Y")
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    fig.tight_layout()
    return fig, ax


def save_attention_boundary_overlay_from_segment_raw(
    segment_raw_path: str | Path,
    union_image_path: str | Path,
    output_path: str | Path,
    *,
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = None,
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
    boundary_color: str = "#d62828",
    boundary_width: int = 3,
) -> Path:
    bbox, points = load_eye_points_from_segment_raw(segment_raw_path)
    comps = _compute_attention_components(
        points=points,
        bbox=bbox,
        max_points=max_points,
        grid_size=grid_size,
        attention_tier=attention_tier,
        gaussian_sigma=gaussian_sigma,
        threshold_quantile=threshold_quantile,
    )
    width = float(comps["width"])
    height = float(comps["height"])
    norm = comps["norm"]
    x_axis = comps["x_axis"]
    y_axis = comps["y_axis"]
    threshold = float(comps["threshold"])

    fig_tmp, ax_tmp = plt.subplots(figsize=(4, 3))
    contour = ax_tmp.contour(x_axis, y_axis, norm, levels=[threshold], colors=["#000000"], linewidths=1.0)
    contour_segments = contour.allsegs[0] if contour.allsegs else []
    plt.close(fig_tmp)

    union_path = Path(union_image_path)
    image = Image.open(union_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_w, image_h = image.size

    for segment in contour_segments:
        if len(segment) < 2:
            continue
        scaled = [
            (
                float(point[0] / width * image_w),
                float(point[1] / height * image_h),
            )
            for point in segment
        ]
        if scaled[0] != scaled[-1]:
            scaled.append(scaled[0])
        draw.line(scaled, fill=boundary_color, width=boundary_width)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def render_attention_contour_from_segment_raw(
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
    bbox, points = load_eye_points_from_segment_raw(segment_raw_path)
    return render_attention_contour(
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


def save_attention_contour_from_segment_raw(
    segment_raw_path: str | Path,
    output_path: str | Path,
    *,
    union_image_path: str | Path | None = None,
    render_mode: str = "boundary_overlay",
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = None,
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
    show_points: bool = True,
    boundary_color: str = "#d62828",
    boundary_width: int = 3,
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 180,
) -> Path:
    if render_mode == "boundary_overlay":
        if union_image_path is None:
            raise ValueError("union_image_path is required when render_mode='boundary_overlay'")
        return save_attention_boundary_overlay_from_segment_raw(
            segment_raw_path,
            union_image_path,
            output_path,
            max_points=max_points,
            grid_size=grid_size,
            attention_tier=attention_tier,
            gaussian_sigma=gaussian_sigma,
            threshold_quantile=threshold_quantile,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )
    if render_mode != "heatmap_contour":
        raise ValueError("render_mode must be one of: boundary_overlay, heatmap_contour")

    fig, _ = render_attention_contour_from_segment_raw(
        segment_raw_path,
        max_points=max_points,
        grid_size=grid_size,
        attention_tier=attention_tier,
        gaussian_sigma=gaussian_sigma,
        threshold_quantile=threshold_quantile,
        show_points=show_points,
        figsize=figsize,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
