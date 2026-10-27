#!/usr/bin/env python3
"""Streamlit UI for browsing cases and running gaze processing."""

from __future__ import annotations
import json
import math
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from processors.gaze.GazeData import GazeData


DEFAULT_CASE = str(Path(__file__).resolve().parent / 'history/case_005')

APP_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = APP_ROOT / "history"


def _discover_history_cases(history_root: Path) -> list[Path]:
    case_dirs = {p.parent for p in history_root.rglob("seg_*_raw.json")}
    return sorted(case_dirs)


def _load_segments(case_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted(case_dir.glob("seg_*_raw.json")):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        seg = raw_path.stem.replace("_raw", "")
        union_png = raw_path.with_name(raw_path.name.replace("_raw.json", "_union.png"))
        union_webp = raw_path.with_name(raw_path.name.replace("_raw.json", "_union.webp"))
        union_path = union_png if union_png.exists() else union_webp
        rows.append(
            {
                "segment": seg,
                "raw_path": raw_path,
                "union_path": union_path,
                "payload": payload,
                "text": (payload.get("text") or "").strip(),
                "start_seconds": payload.get("start_seconds"),
                "end_seconds": payload.get("end_seconds"),
                "eye_points": len(payload.get("eye") or []),
                "mouse_points": len(payload.get("mouse") or []),
                "viewport_boxes": len(payload.get("boxes") or []),
            }
        )
    return rows


def _case_time_label(segments: list[dict[str, Any]]) -> str | None:
    timestamps: list[datetime] = []
    for row in segments:
        for box in row.get("payload", {}).get("boxes") or []:
            if not isinstance(box, dict):
                continue
            value = box.get("timestamp")
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                timestamps.append(dt)
            except ValueError:
                continue
    if not timestamps:
        return None

    # Use earliest available segment timestamp as case-level start hint.
    dt = min(timestamps).astimezone(timezone.utc)
    hour = dt.hour
    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"
    return f"{dt.strftime('%B %-d, %Y')} {period}"


def _ensure_cache_dir() -> Path:
    if "gaze_cache_dir" not in st.session_state:
        st.session_state["gaze_cache_dir"] = tempfile.mkdtemp(prefix="pathlearn_gaze_")
    return Path(st.session_state["gaze_cache_dir"])


def _process_segment(segment_row: dict[str, Any]) -> dict[str, Any]:
    cache_dir = _ensure_cache_dir()
    return GazeData.export_segment_raw_ivt_idt_comparison(
        segment_row["raw_path"],
        output_dir=cache_dir,
        union_image_path=segment_row["union_path"],
        attention_tier="mid",
        canvas_crop_insets=None,
        point_radius=6,
        boundary_width=6,
    )


def _snippet(text: str, limit: int = 52) -> str:
    value = " ".join((text or "").split())
    if not value:
        return "(no text)"
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _map_source_to_union(
    x: float,
    y: float,
    bbox: dict[str, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    px = (float(x) - x1) / bw * width
    py = (float(y) - y1) / bh * height
    return px, py


def _render_unprocessed_overlay(segment_row: dict[str, Any], mode: str) -> str:
    payload = segment_row["payload"]
    bbox = payload.get("boxes_final") or {}
    if not {"x1", "y1", "x2", "y2"}.issubset(bbox.keys()):
        return str(segment_row["union_path"])

    out_dir = _ensure_cache_dir()
    out_path = out_dir / f"{segment_row['segment']}_{mode}_raw_overlay.png"

    image = Image.open(segment_row["union_path"]).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    if mode in {"eye", "all"}:
        for point in payload.get("eye") or []:
            if not isinstance(point, dict) or point.get("x") is None or point.get("y") is None:
                continue
            px, py = _map_source_to_union(point["x"], point["y"], bbox, width, height)
            r = 6
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(40, 200, 150, 140))

    if mode in {"viewports", "all"}:
        boxes = payload.get("boxes") or []
        stride = max(1, math.ceil(len(boxes) / 120))
        for i, box in enumerate(boxes):
            if i % stride != 0:
                continue
            if not isinstance(box, dict):
                continue
            required = ("x1", "y1", "x2", "y2")
            if any(box.get(k) is None for k in required):
                continue
            p1x, p1y = _map_source_to_union(box["x1"], box["y1"], bbox, width, height)
            p2x, p2y = _map_source_to_union(box["x2"], box["y2"], bbox, width, height)
            draw.rectangle((p1x, p1y, p2x, p2y), outline=(255, 196, 0, 180), width=2)

    if mode in {"mouse", "all"}:
        mouse_points = []
        for point in payload.get("mouse") or []:
            if not isinstance(point, dict) or point.get("x") is None or point.get("y") is None:
                continue
            mouse_points.append(_map_source_to_union(point["x"], point["y"], bbox, width, height))
        if len(mouse_points) >= 2:
            draw.line(mouse_points, fill=(80, 120, 255, 180), width=3)
        for px, py in mouse_points[:: max(1, len(mouse_points) // 80)]:
            r = 2
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(80, 120, 255, 180))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)
    return str(out_path)


def _build_raw_summary_contact_sheet(
    *,
    segments: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    cols: int = 3,
    tile_w: int = 400,
    tile_h: int = 400,
) -> dict[str, str]:
    show = list(segments)
    rows = max(1, math.ceil(len(show) / max(cols, 1)))
    out_dir = _ensure_cache_dir()
    out_path = out_dir / "raw_summary_contact_sheet.png"
    pdf_path = out_dir / "raw_summary_contact_sheet.pdf"

    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font_body = _load_font(size=24)

    image_area_w = tile_w
    image_area_h = tile_h
    for idx, row in enumerate(show):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * tile_h

        seg = row["segment"]
        result = results.get(seg)
        if result:
            src_path = Path(result["output_paths"]["processed_boundary"])
        else:
            src_path = Path(row["union_path"])

        src = Image.open(src_path).convert("RGB")
        src.thumbnail((image_area_w, image_area_h), Image.Resampling.LANCZOS)

        pad = Image.new("RGB", (image_area_w, image_area_h), color=(255, 255, 255))
        px = (image_area_w - src.size[0]) // 2
        py = (image_area_h - src.size[1]) // 2
        pad.paste(src, (px, py))
        canvas.paste(pad, (x0, y0))

        # Draw transcript text directly on tile (no background overlay).
        snippet = _snippet(row["text"], limit=100)
        wrapped = textwrap.wrap(snippet, width=38)[:2]
        line_gap = 30
        text_start_y = y0 + int(tile_h * (2.0 / 3.0)) - ((max(1, len(wrapped)) - 1) * line_gap // 2)
        for j, line in enumerate(wrapped):
            bbox = draw.textbbox((0, 0), line, font=font_body, stroke_width=2)
            text_w = bbox[2] - bbox[0]
            tx = x0 + max(0, (tile_w - text_w) // 2)
            ty = text_start_y + j * line_gap
            draw.text(
                (tx, ty),
                line,
                fill=(0, 0, 0, 255),
                font=font_body,
                stroke_width=2,
                stroke_fill=(255, 255, 255, 255),
            )

    canvas.save(out_path, dpi=(220, 220))
    canvas.save(pdf_path, "PDF", resolution=220.0)
    return {"png": str(out_path), "pdf": str(pdf_path)}


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _native_image_width(path: str | Path) -> int | None:
    try:
        with Image.open(path) as im:
            return int(im.size[0])
    except Exception:
        return None


def _show_native_image(path: str | Path, *, caption: str | None = None) -> None:
    width = _native_image_width(path)
    if width is None:
        st.image(str(path), caption=caption)
    else:
        st.image(str(path), caption=caption, width=width)


def main() -> None:
    st.set_page_config(page_title="PathLearn Gaze UI", layout="wide")
    st.markdown(
        """
        <style>
        .app-topbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 48px;
            background: #0f3b57;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 16px;
            font-weight: 700;
            z-index: 1000;
        }
        .block-container {
            padding-top: 62px;
        }
        </style>
        <div class="app-topbar">PathLearn Gaze Case Browser</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Case")
    history_cases = _discover_history_cases(HISTORY_ROOT)
    if not history_cases:
        st.error(f"No history cases found under: {HISTORY_ROOT}")
        return
    labels = [str(p.relative_to(HISTORY_ROOT)) for p in history_cases]
    if "selected_case_label" not in st.session_state or st.session_state["selected_case_label"] not in labels:
        default_label = str(Path(DEFAULT_CASE).relative_to(Path('/Volumes/ExternalHD/PathLearn/videos/example_data')))
        st.session_state["selected_case_label"] = default_label if default_label in labels else labels[0]
    selected_case_label = st.sidebar.selectbox(
        "History folder",
        labels,
        index=labels.index(st.session_state["selected_case_label"]),
    )
    st.sidebar.caption(f"{len(labels)} histories found")
    st.session_state["selected_case_label"] = selected_case_label
    case_dir = HISTORY_ROOT / selected_case_label

    segments = _load_segments(case_dir)
    if not segments:
        st.warning("No seg_*_raw.json files found in this folder.")
        return
    case_time = _case_time_label(segments)

    case_key = str(case_dir.resolve())
    if st.session_state.get("active_case") != case_key:
        st.session_state["active_case"] = case_key
        st.session_state["gaze_results"] = {}
        st.session_state["boundary_ready"] = {}
        st.session_state["selected_item"] = "summary"
        st.session_state["active_mode"] = "none"
        st.session_state["eye_stage"] = {}

    if "gaze_results" not in st.session_state:
        st.session_state["gaze_results"] = {}
    results: dict[str, dict[str, Any]] = st.session_state["gaze_results"]
    if "boundary_ready" not in st.session_state:
        st.session_state["boundary_ready"] = {}
    boundary_ready: dict[str, bool] = st.session_state["boundary_ready"]
    if "eye_stage" not in st.session_state:
        st.session_state["eye_stage"] = {}
    eye_stage: dict[str, str] = st.session_state["eye_stage"]

    seg_names = [row["segment"] for row in segments]
    valid_items = {"summary", *seg_names}
    if st.session_state.get("selected_item") not in valid_items:
        st.session_state["selected_item"] = "summary"

    summary_rows: list[dict[str, Any]] = []
    for row in segments:
        seg = row["segment"]
        result = results.get(seg)
        summary_rows.append(
            {
                "segment": seg,
                "start": row["start_seconds"],
                "end": row["end_seconds"],
                "eye": row["eye_points"],
                "mouse": row["mouse_points"],
                "viewports": row["viewport_boxes"],
                "processed_eye": (result or {}).get("processed_point_count"),
                "fixations": (result or {}).get("fixation_count"),
                "status": "done" if result else "raw",
            }
        )

    # Auto-compute boundary previews for summary if eye data exists.
    for row in segments:
        seg = row["segment"]
        if seg in results:
            boundary_ready[seg] = True
            continue
        if int(row.get("eye_points") or 0) <= 0:
            continue
        try:
            results[seg] = _process_segment(row)
            boundary_ready[seg] = True
        except Exception:
            # Keep summary rendering resilient for partially valid cases.
            pass

    for item in summary_rows:
        seg = item["segment"]
        res = results.get(seg)
        item["processed_eye"] = (res or {}).get("processed_point_count")
        item["fixations"] = (res or {}).get("fixation_count")
        item["status"] = "done" if res else "raw"

    raw_summary_paths = _build_raw_summary_contact_sheet(segments=segments, results=results, cols=3)

    st.sidebar.header("Segments")
    st.sidebar.markdown("**summary**")
    st.sidebar.image(raw_summary_paths["png"], use_column_width=True)
    if st.sidebar.button("Select summary", key="select_summary"):
        st.session_state["selected_item"] = "summary"
        st.session_state["active_mode"] = "none"
    if st.session_state.get("selected_item") == "summary":
        st.sidebar.caption("Selected")
    st.sidebar.markdown("---")

    for idx, row in enumerate(segments, start=1):
        seg = row["segment"]
        start = row.get("start_seconds")
        end = row.get("end_seconds")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            duration = end - start
            title = f"{duration:.2f} s"
        else:
            title = "unknown s"
        st.sidebar.markdown(f"**{title}**")
        st.sidebar.image(str(row["union_path"]), use_column_width=True)
        st.sidebar.caption(f"{idx}. {row['text'] or '(no text)'}")
        if st.sidebar.button("Details", key=f"select_{seg}"):
            st.session_state["selected_item"] = seg
            st.session_state["active_mode"] = "none"
        if st.session_state.get("selected_item") == seg:
            st.sidebar.caption("Selected")
        st.sidebar.markdown("---")

    selected_item = st.session_state["selected_item"]
    if selected_item == "summary":
        st.markdown("### PathLearn History")
        if case_time:
            st.caption(f"Case time: {case_time}")
        else:
            st.caption("Case time: no timestamp available")
        st.image(raw_summary_paths["png"], width=800)
        with open(raw_summary_paths["pdf"], "rb") as handle:
            st.download_button(
                "Download Raw Summary PDF",
                data=handle.read(),
                file_name="raw_summary_contact_sheet.pdf",
                mime="application/pdf",
            )
        st.markdown("### Summary Table")
        st.dataframe(summary_rows, hide_index=True)
        return

    selected_seg = selected_item
    selected_row = next(row for row in segments if row["segment"] == selected_seg)
    selected_boundary_ready = bool(boundary_ready.get(selected_seg, False))
    result = results.get(selected_seg)

    st.subheader(f"Segment Detail: {selected_seg}")
    if st.session_state.get("active_mode") not in {"none", "eye", "viewports", "mouse", "all"}:
        st.session_state["active_mode"] = "none"
    mode_cols = st.columns(4)
    if mode_cols[0].button("Eye"):
        st.session_state["active_mode"] = "eye"
        eye_stage[selected_seg] = "raw"
    if mode_cols[1].button("Viewports"):
        st.session_state["active_mode"] = "viewports"
    if mode_cols[2].button("Mouse"):
        st.session_state["active_mode"] = "mouse"
    if mode_cols[3].button("All"):
        st.session_state["active_mode"] = "all"
    mode = st.session_state["active_mode"]

    raw_eye = _render_unprocessed_overlay(selected_row, "eye")
    raw_viewports = _render_unprocessed_overlay(selected_row, "viewports")
    raw_mouse = _render_unprocessed_overlay(selected_row, "mouse")
    raw_all = _render_unprocessed_overlay(selected_row, "all")

    if mode == "eye":
        eye_col1, eye_col2 = st.columns([1, 1])
        if eye_col1.button("Process gaze"):
            results[selected_seg] = _process_segment(selected_row)
            boundary_ready[selected_seg] = False
            eye_stage[selected_seg] = "processed"
            result = results.get(selected_seg)
            selected_boundary_ready = False
            st.success(f"Processed gaze for {selected_seg}")
        if eye_col2.button("Generate boundary"):
            if selected_seg not in results:
                results[selected_seg] = _process_segment(selected_row)
            boundary_ready[selected_seg] = True
            eye_stage[selected_seg] = "boundary"
            result = results.get(selected_seg)
            selected_boundary_ready = True
            st.success(f"Generated boundary for {selected_seg}")

    if mode == "none":
        _show_native_image(selected_row["union_path"], caption="raw")
    elif mode == "eye":
        stage = eye_stage.get(selected_seg, "raw")
        if stage == "boundary" and result and selected_boundary_ready:
            _show_native_image(result["output_paths"]["processed_boundary"], caption="boundary")
        elif stage == "processed" and result:
            _show_native_image(result["output_paths"]["processed_gaze"], caption="IDT-IVT results")
        else:
            _show_native_image(raw_eye, caption="unprocessed gaze")
    elif mode == "viewports":
        _show_native_image(raw_viewports, caption="unprocessed viewports")
    elif mode == "mouse":
        _show_native_image(raw_mouse, caption="unprocessed mouse")
    elif mode == "all":
        _show_native_image(raw_all, caption="all (unprocessed eye + viewports + mouse)")
    else:
        _show_native_image(selected_row["union_path"], caption="raw")
        st.info(f"{mode} view is not implemented yet.")

    st.markdown(selected_row["text"] or "(no text)")


if __name__ == "__main__":
    main()
