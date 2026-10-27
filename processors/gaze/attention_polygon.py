#!/usr/bin/env python3
"""Gaze attention contour to MultiPolygon conversion utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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

DEFAULT_PROFILE = {
    "outer_target_min": 6,
    "outer_target_max": 9,
    "hole_target_min": 4,
    "hole_target_max": 9,
    "total_vertex_cap": 64,
}

VISUALIZATION_PROFILE = {
    "outer_target_min": 20,
    "outer_target_max": 48,
    "hole_target_min": 8,
    "hole_target_max": 24,
    "total_vertex_cap": 512,
}


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


def load_bbox_and_eye_points(segment_raw_path: str | Path) -> tuple[dict[str, float], list[tuple[float, float]]]:
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


def compute_attention_components(
    *,
    points: list[tuple[float, float]],
    bbox: dict[str, float],
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = None,
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
) -> dict[str, Any]:
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
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox with non-positive size: {bbox}")

    local_points: list[tuple[float, float]] = []
    source_points: list[tuple[float, float]] = []
    for x, y in points:
        if x < x1 or x > x2 or y < y1 or y > y2:
            continue
        local_points.append((x - x1, y - y1))
        source_points.append((x, y))
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
        "source_points": source_points,
        "resolved_tier": resolved_tier,
        "gaussian_sigma": gaussian_sigma,
        "threshold_quantile": threshold_quantile,
    }


def extract_contour_segments(components: dict[str, Any]) -> list[list[tuple[float, float]]]:
    fig_tmp, ax_tmp = plt.subplots(figsize=(4, 3))
    contour = ax_tmp.contour(
        components["x_axis"],
        components["y_axis"],
        components["norm"],
        levels=[float(components["threshold"])],
        colors=["#000000"],
        linewidths=1.0,
    )
    all_segs = contour.allsegs[0] if contour.allsegs else []
    plt.close(fig_tmp)

    segments: list[list[tuple[float, float]]] = []
    for seg in all_segs:
        points = [(float(x), float(y)) for x, y in seg.tolist()]
        if len(points) >= 3:
            segments.append(points)
    return segments


def _close_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    ring = list(points)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _ring_unique_count(ring: list[tuple[float, float]]) -> int:
    if not ring:
        return 0
    return len({(round(x, 9), round(y, 9)) for x, y in ring[:-1]})


def _signed_area(ring: list[tuple[float, float]]) -> float:
    if len(ring) < 4:
        return 0.0
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += (x1 * y2) - (x2 * y1)
    return 0.5 * area


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-9) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= sq_len + eps


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1) if abs(y2 - y1) > 1e-12 else 1e-12
            x_inter = (x2 - x1) * (y - y1) / denom + x1
            if x < x_inter:
                inside = not inside
    return inside


def _segments_intersect(a1, a2, b1, b2, eps: float = 1e-9) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    if abs(o1) <= eps and _point_on_segment(b1[0], b1[1], a1[0], a1[1], a2[0], a2[1]):
        return True
    if abs(o2) <= eps and _point_on_segment(b2[0], b2[1], a1[0], a1[1], a2[0], a2[1]):
        return True
    if abs(o3) <= eps and _point_on_segment(a1[0], a1[1], b1[0], b1[1], b2[0], b2[1]):
        return True
    if abs(o4) <= eps and _point_on_segment(a2[0], a2[1], b1[0], b1[1], b2[0], b2[1]):
        return True
    return False


def _ring_is_self_intersecting(ring: list[tuple[float, float]]) -> bool:
    n = len(ring) - 1
    if n < 3:
        return True
    for i in range(n):
        a1 = ring[i]
        a2 = ring[i + 1]
        for j in range(i + 1, n):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == n - 1:
                continue
            b1 = ring[j]
            b2 = ring[j + 1]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _rdp_open(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    p0 = points[0]
    p1 = points[-1]

    if p0 == p1:
        return points

    max_dist = -1.0
    idx = 0
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    denom = math.hypot(dx, dy)
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if denom <= 1e-12:
            dist = math.hypot(px - p0[0], py - p0[1])
        else:
            dist = abs(dy * px - dx * py + p1[0] * p0[1] - p1[1] * p0[0]) / denom
        if dist > max_dist:
            max_dist = dist
            idx = i

    if max_dist > epsilon:
        left = _rdp_open(points[: idx + 1], epsilon)
        right = _rdp_open(points[idx:], epsilon)
        return left[:-1] + right
    return [p0, p1]


def _simplify_ring_rdp(ring: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(ring) < 5:
        return ring
    open_pts = ring[:-1]
    simplified = _rdp_open(open_pts, epsilon)
    if len(simplified) < 3:
        return ring
    return _close_ring([(float(x), float(y)) for x, y in simplified])


def _adaptive_simplify_ring(
    ring: list[tuple[float, float]],
    *,
    target_min: int,
    target_max: int,
    epsilon_scale: float = 1.0,
) -> list[tuple[float, float]]:
    ring = _close_ring(ring)
    if _ring_unique_count(ring) < 3:
        return ring

    original_vertices = len(ring) - 1
    if original_vertices <= target_max:
        return ring

    xs = [pt[0] for pt in ring[:-1]]
    ys = [pt[1] for pt in ring[:-1]]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    hi = max(diag, 1.0) * max(epsilon_scale, 1e-6)
    lo = 0.0

    best = ring
    best_count = original_vertices
    for _ in range(24):
        mid = (lo + hi) / 2.0
        simplified = _simplify_ring_rdp(ring, epsilon=mid)
        count = len(simplified) - 1

        if count <= target_max:
            if count >= target_min:
                best = simplified
                best_count = count
                hi = mid
            else:
                best = simplified
                best_count = count
                hi = mid
        else:
            lo = mid

    if best_count < 3:
        return ring
    return best


def _resample_ring_to_edge_budget(
    ring: list[tuple[float, float]],
    *,
    max_edges: int,
) -> list[tuple[float, float]]:
    """Force a ring down to <= max_edges while preserving point order."""
    closed = _close_ring(ring)
    open_pts = closed[:-1]
    n = len(open_pts)
    if n <= max_edges or max_edges < 3:
        return closed
    # Evenly sample vertices along the existing contour order.
    idxs = [round(i * (n - 1) / (max_edges - 1)) for i in range(max_edges)]
    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            dedup.append(open_pts[idx])
    if len(dedup) < 3:
        dedup = open_pts[:max(3, min(max_edges, n))]
    return _close_ring(dedup)


def _ring_area_abs(ring: list[tuple[float, float]]) -> float:
    return abs(_signed_area(ring))


def _build_polygon_shells(
    rings: list[list[tuple[float, float]]],
    *,
    roi_area: float,
    min_component_area_ratio: float,
    max_components: int,
) -> list[dict[str, Any]]:
    clean = []
    for ring in rings:
        closed = _close_ring(ring)
        if len(closed) < 4:
            continue
        if _ring_unique_count(closed) < 3:
            continue
        area = _ring_area_abs(closed)
        if area <= 0:
            continue
        if area / max(roi_area, 1e-6) < min_component_area_ratio:
            continue
        if _ring_is_self_intersecting(closed):
            continue
        clean.append({"ring": closed, "area": area})

    if not clean:
        return []

    clean.sort(key=lambda row: row["area"], reverse=True)

    parent = [-1] * len(clean)
    depth = [0] * len(clean)
    for i in range(len(clean)):
        p = clean[i]["ring"][0]
        for j in range(i):
            if _point_in_ring(p, clean[j]["ring"]):
                parent[i] = j
                depth[i] = depth[j] + 1
                break

    outer_indices = [idx for idx in range(len(clean)) if depth[idx] % 2 == 0]
    if max_components > 0:
        outer_indices = outer_indices[:max_components]

    outer_set = set(outer_indices)
    polygons: list[dict[str, Any]] = []
    for outer_idx in outer_indices:
        outer_ring = clean[outer_idx]["ring"]
        holes: list[list[tuple[float, float]]] = []
        for i in range(len(clean)):
            if depth[i] % 2 == 0:
                continue
            p = i
            ancestor = -1
            while p != -1:
                if p in outer_set and depth[p] % 2 == 0:
                    ancestor = p
                    break
                p = parent[p]
            if ancestor == outer_idx:
                holes.append(clean[i]["ring"])
        polygons.append({"outer": outer_ring, "holes": holes})
    return polygons


def _apply_vertex_budget(polygons: list[dict[str, Any]], profile: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for poly in polygons:
        outer = _adaptive_simplify_ring(
            poly["outer"],
            target_min=int(profile["outer_target_min"]),
            target_max=int(profile["outer_target_max"]),
        )
        holes = [
            _adaptive_simplify_ring(
                hole,
                target_min=int(profile["hole_target_min"]),
                target_max=int(profile["hole_target_max"]),
            )
            for hole in poly["holes"]
        ]
        # Hard cap: each shape must stay below 10 edges (user requirement).
        outer = _resample_ring_to_edge_budget(outer, max_edges=int(profile["outer_target_max"]))
        holes = [
            _resample_ring_to_edge_budget(hole, max_edges=int(profile["hole_target_max"]))
            for hole in holes
        ]
        out.append({"outer": outer, "holes": holes})

    cap = int(profile["total_vertex_cap"])

    def _total_vertices(items: list[dict[str, Any]]) -> int:
        total = 0
        for row in items:
            total += max(0, len(row["outer"]) - 1)
            for hole in row["holes"]:
                total += max(0, len(hole) - 1)
        return total

    total = _total_vertices(out)
    if total <= cap:
        return out

    scale = 1.2
    for _ in range(8):
        candidate = []
        for poly in polygons:
            outer = _adaptive_simplify_ring(
                poly["outer"],
                target_min=3,
                target_max=int(profile["outer_target_max"]),
                epsilon_scale=scale,
            )
            holes = [
                _adaptive_simplify_ring(
                    hole,
                    target_min=3,
                    target_max=int(profile["hole_target_max"]),
                    epsilon_scale=scale,
                )
                for hole in poly["holes"]
            ]
            candidate.append({"outer": outer, "holes": holes})
        total = _total_vertices(candidate)
        out = candidate
        if total <= cap:
            break
        scale *= 1.35
    return out


def _ring_to_roi_norm(ring: list[tuple[float, float]], width: float, height: float) -> list[list[float]]:
    out = []
    w = max(width, 1e-6)
    h = max(height, 1e-6)
    for x, y in ring:
        xn = min(max(float(x) / w, 0.0), 1.0)
        yn = min(max(float(y) / h, 0.0), 1.0)
        out.append([round(xn, 6), round(yn, 6)])
    return out


def _ring_to_source_int(ring: list[tuple[float, float]], bbox: dict[str, float]) -> list[list[int]]:
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])
    out = []
    for x, y in ring:
        sx = int(round(x1 + float(x)))
        sy = int(round(y1 + float(y)))
        sx = min(max(sx, int(math.floor(x1))), int(math.ceil(x2)))
        sy = min(max(sy, int(math.floor(y1))), int(math.ceil(y2)))
        out.append([sx, sy])
    return out


def build_multipolygon_from_components(
    components: dict[str, Any],
    *,
    bbox: dict[str, float],
    min_component_area_ratio: float = 0.003,
    max_components: int = 3,
    profile: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = dict(DEFAULT_PROFILE if profile is None else profile)
    rings = extract_contour_segments(components)
    roi_area = float(components["width"]) * float(components["height"])
    polygons = _build_polygon_shells(
        rings,
        roi_area=roi_area,
        min_component_area_ratio=min_component_area_ratio,
        max_components=max_components,
    )
    polygons = _apply_vertex_budget(polygons, profile)

    roi_polygons = []
    source_polygons = []
    for poly in polygons:
        outer = _close_ring(poly["outer"])
        holes = [_close_ring(hole) for hole in poly["holes"]]
        roi_polygons.append(
            {
                "outer": _ring_to_roi_norm(outer, float(components["width"]), float(components["height"])),
                "holes": [
                    _ring_to_roi_norm(hole, float(components["width"]), float(components["height"]))
                    for hole in holes
                ],
            }
        )
        source_polygons.append(
            {
                "outer": _ring_to_source_int(outer, bbox),
                "holes": [_ring_to_source_int(hole, bbox) for hole in holes],
            }
        )

    stats = {
        "raw_contour_count": len(rings),
        "component_count_kept": len(polygons),
        "raw_contour_vertex_count": sum(max(0, len(seg) - 1) for seg in rings),
        "simplified_vertex_count": sum(
            max(0, len(poly["outer"]) - 1)
            + sum(max(0, len(hole) - 1) for hole in poly["holes"])
            for poly in source_polygons
        ),
    }

    return (
        {
            "type": "MultiPolygon",
            "polygons": roi_polygons,
        },
        {
            "type": "MultiPolygon",
            "polygons": source_polygons,
        },
        stats,
    )


def build_segment_gaze_multipolygon_artifact(
    segment_raw_path: str | Path,
    *,
    image_file_name: str,
    case_id: str | None = None,
    slide_id: str | None = None,
    segment_key: str | None = None,
    max_points: int = 5000,
    grid_size: int = 256,
    attention_tier: str | None = "mid",
    gaussian_sigma: float | None = None,
    threshold_quantile: float | None = None,
    min_component_area_ratio: float = 0.003,
    max_components: int = 3,
) -> dict[str, Any]:
    bbox, points = load_bbox_and_eye_points(segment_raw_path)
    components = compute_attention_components(
        points=points,
        bbox=bbox,
        max_points=max_points,
        grid_size=grid_size,
        attention_tier=attention_tier,
        gaussian_sigma=gaussian_sigma,
        threshold_quantile=threshold_quantile,
    )
    resolved_tier = str(components["resolved_tier"])

    multipolygon_roi_norm, multipolygon_source_wsi_px, stats = build_multipolygon_from_components(
        components,
        bbox=bbox,
        min_component_area_ratio=min_component_area_ratio,
        max_components=max_components,
        profile=DEFAULT_PROFILE,
    )
    vis_roi_norm, vis_source_wsi_px, vis_stats = build_multipolygon_from_components(
        components,
        bbox=bbox,
        min_component_area_ratio=min_component_area_ratio,
        max_components=max_components,
        profile=VISUALIZATION_PROFILE,
    )

    roi_rect_source_wsi_px = {
        "x1": int(round(float(bbox["x1"]))),
        "y1": int(round(float(bbox["y1"]))),
        "x2": int(round(float(bbox["x2"]))),
        "y2": int(round(float(bbox["y2"]))),
    }

    source_points = [
        [int(round(float(x))), int(round(float(y)))] for x, y in components["source_points"]
    ]
    width = float(components["width"])
    height = float(components["height"])
    roi_points = [
        [
            round(min(max(float(x) / max(width, 1e-6), 0.0), 1.0), 6),
            round(min(max(float(y) / max(height, 1e-6), 0.0), 1.0), 6),
        ]
        for x, y in components["local_points"]
    ]

    artifact = {
        "image_file_name": image_file_name,
        "case_id": case_id,
        "slide_id": slide_id,
        "segment_key": segment_key,
        "roi_rect_source_wsi_px": roi_rect_source_wsi_px,
        "points_source_wsi_px": source_points,
        "points_roi_norm": roi_points,
        "multipolygon_source_wsi_px": multipolygon_source_wsi_px,
        "multipolygon_roi_norm": multipolygon_roi_norm,
        "multipolygon_lod_source_wsi_px": {
            "training": multipolygon_source_wsi_px,
            "visualization": vis_source_wsi_px,
        },
        "multipolygon_lod_roi_norm": {
            "training": multipolygon_roi_norm,
            "visualization": vis_roi_norm,
        },
        "algorithm": {
            "name": "gaze_attention_contour_to_multipolygon",
            "version": "v2",
            "attention_tier": resolved_tier,
            "grid_size": int(grid_size),
            "gaussian_sigma": float(components["gaussian_sigma"]),
            "threshold_quantile": float(components["threshold_quantile"]),
            "vertex_profile": "aggressive_compression",
            **DEFAULT_PROFILE,
            "lod_profiles": {
                "training": dict(DEFAULT_PROFILE),
                "visualization": dict(VISUALIZATION_PROFILE),
            },
        },
        "quality": stats,
        "quality_lods": {
            "training": stats,
            "visualization": vis_stats,
        },
    }
    validate_gaze_multipolygon_artifact(artifact)
    return artifact


def _validate_ring(
    ring: list[list[float | int]],
    *,
    normalized: bool,
    enforce_topology: bool,
) -> None:
    if len(ring) < 4:
        raise ValueError("Ring must contain at least 4 points including closure.")
    if ring[0] != ring[-1]:
        raise ValueError("Ring must be closed (first point equals last point).")

    points = []
    for pt in ring:
        if not isinstance(pt, list) or len(pt) != 2:
            raise ValueError("Each point must be [x,y].")
        x, y = pt
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("Point coordinates must be numeric.")
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Point coordinates must be finite.")
        if normalized and not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ValueError("Normalized coordinates must be within [0,1].")
        points.append((float(x), float(y)))

    closed = _close_ring(points)
    if _ring_unique_count(closed) < 3:
        raise ValueError("Ring must contain at least 3 unique points.")
    if _ring_area_abs(closed) <= 1e-8:
        raise ValueError("Ring area must be non-zero.")
    if enforce_topology and _ring_is_self_intersecting(closed):
        raise ValueError("Ring cannot self-intersect.")


def _validate_multipolygon_shape(
    payload: dict[str, Any],
    *,
    normalized: bool,
    enforce_topology: bool,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("MultiPolygon payload must be an object.")
    if payload.get("type") != "MultiPolygon":
        raise ValueError("MultiPolygon type must be 'MultiPolygon'.")
    polygons = payload.get("polygons")
    if not isinstance(polygons, list):
        raise ValueError("MultiPolygon.polygons must be a list.")
    for poly in polygons:
        if not isinstance(poly, dict):
            raise ValueError("Each polygon must be an object.")
        outer = poly.get("outer")
        holes = poly.get("holes")
        if not isinstance(outer, list) or not isinstance(holes, list):
            raise ValueError("Polygon must include outer(list) and holes(list).")
        _validate_ring(outer, normalized=normalized, enforce_topology=enforce_topology)
        outer_ring = [(float(x), float(y)) for x, y in outer]
        for hole in holes:
            _validate_ring(hole, normalized=normalized, enforce_topology=enforce_topology)
            hole_ring = [(float(x), float(y)) for x, y in hole]
            if not _point_in_ring(hole_ring[0], outer_ring):
                raise ValueError("Hole must lie inside outer ring.")


def validate_gaze_multipolygon_artifact(payload: dict[str, Any]) -> None:
    required = [
        "image_file_name",
        "case_id",
        "slide_id",
        "segment_key",
        "roi_rect_source_wsi_px",
        "points_source_wsi_px",
        "points_roi_norm",
        "multipolygon_source_wsi_px",
        "multipolygon_roi_norm",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required field: {key}")

    image_file_name = payload.get("image_file_name")
    if not isinstance(image_file_name, str) or not image_file_name.strip():
        raise ValueError("image_file_name must be a non-empty string.")

    segment_key = payload.get("segment_key")
    if segment_key is not None and (not isinstance(segment_key, str) or not segment_key.strip()):
        raise ValueError("segment_key must be null or a non-empty string.")

    roi = payload.get("roi_rect_source_wsi_px")
    if not isinstance(roi, dict):
        raise ValueError("roi_rect_source_wsi_px must be an object.")
    for k in ("x1", "y1", "x2", "y2"):
        v = roi.get(k)
        if not isinstance(v, int):
            raise ValueError("roi_rect_source_wsi_px must use integer x1,y1,x2,y2.")
    if not (roi["x1"] < roi["x2"] and roi["y1"] < roi["y2"]):
        raise ValueError("roi_rect_source_wsi_px requires x1<x2 and y1<y2.")

    points_src = payload.get("points_source_wsi_px")
    points_norm = payload.get("points_roi_norm")
    if not isinstance(points_src, list) or not isinstance(points_norm, list):
        raise ValueError("points_source_wsi_px and points_roi_norm must be lists.")
    if len(points_src) != len(points_norm):
        raise ValueError("points_source_wsi_px and points_roi_norm must be length-matched.")

    for p in points_src:
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(v, int) for v in p):
            raise ValueError("points_source_wsi_px must contain [int,int].")

    for p in points_norm:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("points_roi_norm must contain [x,y].")
        x, y = p
        if not isinstance(x, (float, int)) or not isinstance(y, (float, int)):
            raise ValueError("points_roi_norm must contain numeric coordinates.")
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ValueError("points_roi_norm coordinates must be within [0,1].")

    mp_src = payload.get("multipolygon_source_wsi_px")
    mp_norm = payload.get("multipolygon_roi_norm")
    _validate_multipolygon_shape(mp_src, normalized=False, enforce_topology=False)
    _validate_multipolygon_shape(mp_norm, normalized=True, enforce_topology=True)

    src_polys = mp_src.get("polygons")
    norm_polys = mp_norm.get("polygons")
    if len(src_polys) != len(norm_polys):
        raise ValueError("Polygon counts must match between source and normalized outputs.")

    for src_poly, norm_poly in zip(src_polys, norm_polys):
        if len(src_poly["outer"]) != len(norm_poly["outer"]):
            raise ValueError("Outer ring point counts must match between coordinate spaces.")
        if len(src_poly["holes"]) != len(norm_poly["holes"]):
            raise ValueError("Hole counts must match between coordinate spaces.")
        for src_hole, norm_hole in zip(src_poly["holes"], norm_poly["holes"]):
            if len(src_hole) != len(norm_hole):
                raise ValueError("Hole point counts must match between coordinate spaces.")

    lod_src = payload.get("multipolygon_lod_source_wsi_px")
    lod_norm = payload.get("multipolygon_lod_roi_norm")
    if lod_src is not None or lod_norm is not None:
        if not isinstance(lod_src, dict) or not isinstance(lod_norm, dict):
            raise ValueError("multipolygon_lod_source_wsi_px and multipolygon_lod_roi_norm must be objects.")
        required_lods = ("training", "visualization")
        for lod_name in required_lods:
            if lod_name not in lod_src or lod_name not in lod_norm:
                raise ValueError(f"Missing LOD '{lod_name}' in multipolygon LOD payload.")
            _validate_multipolygon_shape(
                lod_src[lod_name], normalized=False, enforce_topology=False
            )
            _validate_multipolygon_shape(
                lod_norm[lod_name], normalized=True, enforce_topology=True
            )


def write_gaze_multipolygon_artifact(payload: dict[str, Any], output_path: str | Path) -> Path:
    validate_gaze_multipolygon_artifact(payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return out
