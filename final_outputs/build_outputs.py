#!/usr/bin/env python3
"""Build PathLearn final outputs from processed segment dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from final_outputs.gaze_attention import attention_tier_presets, save_attention_contour_from_segment_raw


@dataclass
class SegmentRecord:
    segment_id: int
    segment_key: str
    raw_path: Path
    text_path: Path
    union_path: Path
    all_path: Path
    eye_path: Path
    mouse_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final outputs from processed text-image-eye-tracking segments: "
            "review PDF, grounded JSONL dataset, and QA benchmark."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="history/case_003",
        help="Processed segment folder with seg_00xx_* files",
    )
    parser.add_argument(
        "--output-dir",
        default="final_outputs/example_data",
        help="Output directory",
    )
    parser.add_argument(
        "--case-context-json",
        default=None,
        help=(
            "Case background JSON (organ, suspected_diagnosis, patient_age, clinical_context, "
            "optional case_thumbnail). Required by default for meaningful generation."
        ),
    )
    parser.add_argument(
        "--background-text",
        default=None,
        help="Single flexible case background text (human-written hint or finalized background)",
    )
    parser.add_argument(
        "--auto-background",
        action="store_true",
        help="Auto-generate background_text from 3x3 union collage + transcript using Gemini",
    )
    parser.add_argument(
        "--background-model",
        default="gemini-3-flash-preview",
        help="Gemini model for background generation",
    )
    parser.add_argument(
        "--background-grid-size",
        type=int,
        default=3,
        help="Grid size for collage (3 means 3x3)",
    )
    parser.add_argument(
        "--background-sample-count",
        type=int,
        default=9,
        help="How many union screenshots to sample for collage",
    )
    parser.add_argument(
        "--allow-missing-case-context",
        action="store_true",
        help="Allow generation without case context metadata (not recommended)",
    )
    parser.add_argument(
        "--context-window-size",
        type=int,
        default=5,
        help="Number of previous transcript segments to include as context window",
    )
    parser.add_argument(
        "--transcript-json",
        default="videos/example_data/brain/transcription.json",
        help="Transcript JSON used to generate QA benchmark",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip review PDF generation",
    )
    parser.add_argument(
        "--skip-grounded",
        action="store_true",
        help="Skip text-image-grounding JSONL generation",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA benchmark generation",
    )
    parser.add_argument(
        "--qa-provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="Provider for QA generation",
    )
    parser.add_argument(
        "--qa-model",
        default="gemini-3-flash-preview",
        help="Model used to generate QA benchmark",
    )
    parser.add_argument(
        "--qa-count",
        type=int,
        default=25,
        help="Number of QA questions to generate",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Provider API key override (OPENAI_API_KEY or GEMINI_API_KEY if omitted)",
    )
    parser.add_argument(
        "--qa-retry-seconds",
        type=int,
        default=5,
        help="Wait seconds before retry on provider rate limits",
    )
    parser.add_argument(
        "--qa-max-retries",
        type=int,
        default=6,
        help="Max retries for a single failed QA generation call",
    )
    parser.add_argument(
        "--refine-text",
        action="store_true",
        help="Use Gemini to refine segment text for human/machine outputs",
    )
    parser.add_argument(
        "--refine-model",
        default="gemini-3-flash-preview",
        help="Gemini model for refinement",
    )
    parser.add_argument(
        "--refine-max-items",
        type=int,
        default=0,
        help="Limit refinement to first N segments (0 means all)",
    )
    parser.add_argument(
        "--refine-retry-seconds",
        type=int,
        default=5,
        help="Wait seconds before retry on Gemini refinement rate limits",
    )
    parser.add_argument(
        "--refine-max-retries",
        type=int,
        default=6,
        help="Max retries for a single failed refinement call",
    )
    parser.add_argument(
        "--skip-eye-attention",
        action="store_true",
        help="Skip generation of eye-attention contour images from seg_*_raw.json",
    )
    parser.add_argument(
        "--eye-attention-tier",
        choices=sorted(attention_tier_presets().keys()),
        default="low",
        help="Preset tier for eye-attention contour size",
    )
    parser.add_argument(
        "--eye-attention-grid-size",
        type=int,
        default=256,
        help="Grid size for eye-attention contour rasterization",
    )
    parser.add_argument(
        "--eye-attention-sigma",
        type=float,
        default=None,
        help="Override gaussian sigma for eye-attention contour (optional)",
    )
    parser.add_argument(
        "--eye-attention-quantile",
        type=float,
        default=None,
        help="Override threshold quantile for eye-attention contour (optional)",
    )
    parser.add_argument(
        "--eye-attention-show-points",
        action="store_true",
        help="Overlay raw eye points on eye-attention contour images",
    )
    parser.add_argument(
        "--eye-attention-render-mode",
        choices=["boundary_overlay", "heatmap_contour"],
        default="boundary_overlay",
        help="How to render eye-attention images",
    )
    parser.add_argument(
        "--eye-attention-boundary-color",
        default="#d62828",
        help="Boundary color when --eye-attention-render-mode=boundary_overlay",
    )
    parser.add_argument(
        "--eye-attention-boundary-width",
        type=int,
        default=3,
        help="Boundary width in pixels when --eye-attention-render-mode=boundary_overlay",
    )
    return parser.parse_args()


def discover_segments(dataset_root: Path) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    for raw_path in sorted(dataset_root.glob("seg_*_raw.json")):
        name = raw_path.name
        seg_key = name.split("_")[1]
        records.append(
            SegmentRecord(
                segment_id=int(seg_key),
                segment_key=seg_key,
                raw_path=raw_path,
                text_path=dataset_root / f"seg_{seg_key}_text.txt",
                union_path=dataset_root / f"seg_{seg_key}_union.png",
                all_path=dataset_root / f"seg_{seg_key}_all.png",
                eye_path=dataset_root / f"seg_{seg_key}_eye.png",
                mouse_path=dataset_root / f"seg_{seg_key}_mouse.png",
            )
        )
    return records


def load_case_context(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Case context must be a JSON object: {path}")
    return payload


def validate_case_context(case_context: dict[str, Any] | None, allow_missing: bool) -> None:
    if case_context is None:
        if allow_missing:
            return
        raise ValueError(
            "Missing case background metadata. Pass --case-context-json or use "
            "--allow-missing-case-context to bypass."
        )
    required = ["organ", "suspected_diagnosis", "patient_age", "clinical_context"]
    missing = [k for k in required if not str(case_context.get(k, "")).strip()]
    if missing and not allow_missing:
        raise ValueError(
            f"Case context missing required fields: {missing}. "
            "Provide them in --case-context-json or use --allow-missing-case-context."
        )


def sample_records_for_collage(records: list[SegmentRecord], sample_count: int) -> list[SegmentRecord]:
    if not records:
        return []
    if sample_count <= 0 or sample_count >= len(records):
        return records
    if sample_count == 1:
        return [records[len(records) // 2]]
    idxs = []
    n = len(records)
    for i in range(sample_count):
        idx = round(i * (n - 1) / (sample_count - 1))
        idxs.append(idx)
    # preserve order, dedupe
    seen = set()
    selected = []
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            selected.append(records[idx])
    return selected


def build_union_collage(
    records: list[SegmentRecord],
    output_path: Path,
    *,
    grid_size: int,
) -> Path:
    selected = sample_records_for_collage(records, grid_size * grid_size)
    if not selected:
        raise ValueError("No records available to build collage.")
    cell = 640
    canvas = Image.new("RGB", (grid_size * cell, grid_size * cell), color=(245, 245, 245))
    for i in range(grid_size * grid_size):
        x = (i % grid_size) * cell
        y = (i // grid_size) * cell
        if i >= len(selected):
            continue
        img = Image.open(selected[i].union_path).convert("RGB")
        img.thumbnail((cell, cell))
        # center in cell
        dx = x + (cell - img.width) // 2
        dy = y + (cell - img.height) // 2
        canvas.paste(img, (dx, dy))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def collect_segment_transcript(records: list[SegmentRecord]) -> str:
    parts = []
    for r in records:
        text = _read_segment_text(r)
        if text:
            parts.append(f"[seg_{r.segment_key}] {text}")
    return "\n".join(parts)


def compose_background_seed(case_context: dict[str, Any] | None, arg_background_text: str | None) -> str:
    if arg_background_text and arg_background_text.strip():
        return arg_background_text.strip()
    if not case_context:
        return ""
    if case_context.get("background_text"):
        return str(case_context.get("background_text")).strip()
    bits = []
    if case_context.get("organ"):
        bits.append(f"Organ: {case_context.get('organ')}")
    if case_context.get("suspected_diagnosis"):
        bits.append(f"Suspected diagnosis: {case_context.get('suspected_diagnosis')}")
    if case_context.get("patient_age"):
        bits.append(f"Patient age: {case_context.get('patient_age')}")
    if case_context.get("clinical_context"):
        bits.append(f"Clinical context: {case_context.get('clinical_context')}")
    return ". ".join(bits).strip()


def generate_background_with_gemini(
    *,
    collage_path: Path,
    segment_transcript: str,
    seed_background: str,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for background generation. Install with: pip install google-genai"
        ) from exc
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.")

    prompt = (
        "You are summarizing pathology case background from sampled whole-slide screenshots and transcript snippets.\n"
        "Return strict JSON only:\n"
        "- background_text (string)\n"
        "- confidence (number 0..1)\n"
        "- uncertainty_note (string)\n\n"
        "Rules:\n"
        "1) background_text should be concise (2-5 sentences) and useful for downstream data generation.\n"
        "2) Mention organ and likely diagnostic context only if reasonably supported.\n"
        "3) If uncertain, explicitly state uncertainty in uncertainty_note.\n"
        "4) Do not hallucinate patient history.\n\n"
        f"Seed background (human hint, may be partial): {seed_background if seed_background else '(none)'}\n\n"
        f"Transcript segments:\n{segment_transcript}\n"
    )
    client = genai.Client(api_key=api_key)

    def _invoke():
        with collage_path.open("rb") as fh:
            image_bytes = fh.read()
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

    response = _call_with_retry(_invoke, retries=retries, retry_seconds=retry_seconds)
    text_response = getattr(response, "text", None)
    if not text_response:
        raise ValueError("Gemini returned empty background response.")
    parsed = json.loads(text_response)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini returned non-object JSON for background generation.")
    return parsed


def resolve_background_text(
    *,
    case_context: dict[str, Any] | None,
    arg_background_text: str | None,
    auto_background: bool,
    records: list[SegmentRecord],
    output_dir: Path,
    transcript_json: Path,
    background_model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
    grid_size: int,
    sample_count: int,
    allow_missing_case_context: bool,
) -> tuple[str | None, dict[str, Any]]:
    seed = compose_background_seed(case_context, arg_background_text)
    info: dict[str, Any] = {"source": "seed", "seed_background": seed}
    if auto_background:
        selected = sample_records_for_collage(records, sample_count)
        collage_path = build_union_collage(
            selected,
            output_dir / "background_collage.png",
            grid_size=grid_size,
        )
        transcript_text = load_transcript_text(transcript_json)
        if not transcript_text:
            transcript_text = collect_segment_transcript(records)
        payload = generate_background_with_gemini(
            collage_path=collage_path,
            segment_transcript=transcript_text,
            seed_background=seed,
            model=background_model,
            api_key=api_key,
            retries=retries,
            retry_seconds=retry_seconds,
        )
        background_text = str(payload.get("background_text", "")).strip()
        if not background_text:
            if allow_missing_case_context:
                return None, {"source": "auto_empty", "payload": payload}
            raise ValueError("Auto background generation returned empty background_text.")
        info = {
            "source": "auto_gemini",
            "seed_background": seed,
            "payload": payload,
            "collage_path": str(collage_path),
        }
        return background_text, info

    if seed:
        return seed, info
    if allow_missing_case_context:
        return None, {"source": "missing_allowed"}
    raise ValueError(
        "No background_text available. Provide --background-text, --case-context-json with details, "
        "or use --auto-background."
    )


def build_segment_text_lookup(records: list[SegmentRecord]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for r in records:
        raw = json.loads(r.raw_path.read_text(encoding="utf-8"))
        text = r.text_path.read_text(encoding="utf-8").strip() if r.text_path.exists() else str(raw.get("text", "")).strip()
        lookup[r.segment_key] = text
    return lookup


def prior_context_text(
    records: list[SegmentRecord],
    current_idx: int,
    *,
    segment_text_lookup: dict[str, str],
    window_size: int,
) -> str:
    start = max(0, current_idx - window_size)
    context_parts: list[str] = []
    for i in range(start, current_idx):
        rk = records[i].segment_key
        txt = segment_text_lookup.get(rk, "").strip()
        if txt:
            context_parts.append(f"[seg_{rk}] {txt}")
    return "\n".join(context_parts)


def build_eye_attention_images(
    records: list[SegmentRecord],
    *,
    output_dir: Path,
    repo_root: Path,
    attention_tier: str,
    grid_size: int,
    sigma: float | None,
    quantile: float | None,
    show_points: bool,
    render_mode: str,
    boundary_color: str,
    boundary_width: int,
) -> dict[str, str]:
    eye_dir = output_dir / "eye_attention"
    eye_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for record in records:
        mode_tag = "boundary" if render_mode == "boundary_overlay" else "heatmap"
        output_path = eye_dir / f"seg_{record.segment_key}_eye_attention_{mode_tag}_{attention_tier}.png"
        try:
            save_attention_contour_from_segment_raw(
                record.raw_path,
                output_path,
                union_image_path=record.union_path,
                render_mode=render_mode,
                grid_size=grid_size,
                attention_tier=attention_tier,
                gaussian_sigma=sigma,
                threshold_quantile=quantile,
                show_points=show_points,
                boundary_color=boundary_color,
                boundary_width=boundary_width,
            )
        except Exception as exc:
            print(f"eye_attention_skip segment={record.segment_key} reason={type(exc).__name__}:{exc}")
            continue
        mapping[record.segment_key] = str(output_path.relative_to(repo_root))
    return mapping


def export_grounded_jsonl(
    records: list[SegmentRecord],
    output_path: Path,
    *,
    repo_root: Path,
    case_context: dict[str, Any] | None,
    background_text: str | None,
    background_info: dict[str, Any] | None,
    context_window_size: int,
    refinement_map: dict[str, dict[str, Any]] | None,
    eye_attention_map: dict[str, str] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_text_lookup = build_segment_text_lookup(records)
    key_to_idx = {r.segment_key: idx for idx, r in enumerate(records)}
    with output_path.open("w", encoding="utf-8") as handle:
        for r in records:
            raw = json.loads(r.raw_path.read_text(encoding="utf-8"))
            text = r.text_path.read_text(encoding="utf-8").strip() if r.text_path.exists() else raw.get("text", "")
            idx = key_to_idx[r.segment_key]
            context_text = prior_context_text(
                records,
                idx,
                segment_text_lookup=segment_text_lookup,
                window_size=context_window_size,
            )
            row = {
                "segment_id": r.segment_id,
                "segment_key": r.segment_key,
                "start_seconds": raw.get("start_seconds"),
                "end_seconds": raw.get("end_seconds"),
                "text": (
                    (refinement_map or {}).get(r.segment_key, {}).get("text_refined", text)
                ),
                "text_raw": text,
                "text_refined": (refinement_map or {}).get(r.segment_key, {}).get("text_refined"),
                "text_refinement": (refinement_map or {}).get(r.segment_key, {}),
                "text_prior_context": context_text,
                "images": {
                    "union": str(r.union_path.relative_to(repo_root)),
                    "all": str(r.all_path.relative_to(repo_root)),
                    "eye": str(r.eye_path.relative_to(repo_root)),
                    "mouse": str(r.mouse_path.relative_to(repo_root)),
                    "eye_attention": (eye_attention_map or {}).get(r.segment_key),
                },
                "grounding": {
                    "boxes": raw.get("boxes"),
                    "boxes_final": raw.get("boxes_final"),
                    "boxes_final_source": raw.get("boxes_final_source"),
                    "boxes_final_is_square": raw.get("boxes_final_is_square"),
                    "eye_points": raw.get("eye"),
                    "mouse_points": raw.get("mouse"),
                },
                "case_context": case_context or {},
                "background_text": background_text,
                "background_info": background_info or {},
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return output_path


def export_human_readable_jsonl(
    records: list[SegmentRecord],
    output_path: Path,
    *,
    repo_root: Path,
    case_context: dict[str, Any] | None,
    background_text: str | None,
    background_info: dict[str, Any] | None,
    context_window_size: int,
    refinement_map: dict[str, dict[str, Any]] | None,
    eye_attention_map: dict[str, str] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_text_lookup = build_segment_text_lookup(records)
    key_to_idx = {r.segment_key: idx for idx, r in enumerate(records)}
    with output_path.open("w", encoding="utf-8") as handle:
        for r in records:
            raw = json.loads(r.raw_path.read_text(encoding="utf-8"))
            text_raw = r.text_path.read_text(encoding="utf-8").strip() if r.text_path.exists() else raw.get("text", "")
            idx = key_to_idx[r.segment_key]
            context_text = prior_context_text(
                records,
                idx,
                segment_text_lookup=segment_text_lookup,
                window_size=context_window_size,
            )
            refined = (refinement_map or {}).get(r.segment_key, {}).get("text_refined")
            row = {
                "segment_id": r.segment_id,
                "segment_key": r.segment_key,
                "start_seconds": raw.get("start_seconds"),
                "end_seconds": raw.get("end_seconds"),
                "image_union": str(r.union_path.relative_to(repo_root)),
                "image_eye_attention": (eye_attention_map or {}).get(r.segment_key),
                "text_raw": text_raw,
                "text_refined": refined,
                "text": refined or text_raw,
                "text_prior_context": context_text,
                "text_refinement": (refinement_map or {}).get(r.segment_key, {}),
                "case_context": case_context or {},
                "background_text": background_text,
                "background_info": background_info or {},
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return output_path


def export_review_pdf(
    records: list[SegmentRecord],
    output_path: Path,
    *,
    case_context: dict[str, Any] | None,
    background_text: str | None,
    context_window_size: int,
    refinement_map: dict[str, dict[str, Any]] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_text_lookup = build_segment_text_lookup(records)
    key_to_idx = {r.segment_key: idx for idx, r in enumerate(records)}
    with PdfPages(output_path) as pdf:
        for r in records:
            raw = json.loads(r.raw_path.read_text(encoding="utf-8"))
            text = r.text_path.read_text(encoding="utf-8").strip() if r.text_path.exists() else raw.get("text", "")
            text_refined = (refinement_map or {}).get(r.segment_key, {}).get("text_refined")
            idx = key_to_idx[r.segment_key]
            context_text = prior_context_text(
                records,
                idx,
                segment_text_lookup=segment_text_lookup,
                window_size=context_window_size,
            )
            image = mpimg.imread(r.union_path)

            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])

            ax_img = fig.add_subplot(gs[0])
            ax_img.imshow(image)
            ax_img.axis("off")
            ax_img.set_title(
                f"Segment {r.segment_key} | {raw.get('start_seconds')}s - {raw.get('end_seconds')}s",
                fontsize=12,
            )

            ax_text = fig.add_subplot(gs[1])
            ax_text.axis("off")
            context_header = ""
            if case_context:
                context_header = (
                    f"Organ: {case_context.get('organ')} | "
                    f"Suspected Dx: {case_context.get('suspected_diagnosis')} | "
                    f"Patient Age: {case_context.get('patient_age')}\n"
                )
            ax_text.text(
                0.0,
                0.8,
                (
                    f"{context_header}"
                    f"Background: {background_text if background_text else '(none)'}\n\n"
                    f"Current segment raw text: {text if text else '(empty text)'}\n\n"
                    f"Refined text: {text_refined if text_refined else '(not refined)'}\n\n"
                    f"Prior context (last {context_window_size}):\n"
                    f"{context_text if context_text else '(none)'}"
                ),
                fontsize=9.5,
                va="top",
                wrap=True,
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def load_transcript_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("text"):
            return str(payload["text"]).strip()
        segments = payload.get("segments")
        if isinstance(segments, list):
            return " ".join(str(seg.get("text", "")).strip() for seg in segments if isinstance(seg, dict)).strip()
    if isinstance(payload, list):
        return " ".join(str(seg.get("text", "")).strip() for seg in payload if isinstance(seg, dict)).strip()
    return ""


def generate_qa_with_openai(
    transcript_text: str,
    *,
    question_count: int,
    model: str,
    api_key: str | None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for QA generation. Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are generating pathology education benchmark questions.\n"
        f"Create exactly {question_count} multiple-choice questions grounded in the transcript.\n"
        "Output strict JSON with one top-level object key: 'questions'.\n"
        "Each question item must include:\n"
        "- id (int)\n"
        "- question (string)\n"
        "- options (array of exactly 4 strings)\n"
        "- answer_index (0-3)\n"
        "- explanation (string)\n"
        "- evidence_text (short quote/paraphrase from transcript)\n\n"
        "Transcript:\n"
        f"{transcript_text}\n"
    )

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. Do not include markdown. "
                    "Questions must test diagnostic reasoning, not trivial wording."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("QA generation returned non-dict JSON payload.")
    return parsed


def _read_segment_text(record: SegmentRecord) -> str:
    if record.text_path.exists():
        text = record.text_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raw = json.loads(record.raw_path.read_text(encoding="utf-8"))
    return str(raw.get("text", "")).strip()


def _segment_is_trivially_unusable(text: str) -> bool:
    words = [w for w in text.split() if w.strip()]
    if len(words) < 6:
        return True
    if len(text) < 30:
        return True
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "503" in msg
        or "unavailable" in msg
        or "high demand" in msg
    )


def _call_with_retry(fn, *, retries: int, retry_seconds: int):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            if attempt > retries or not _is_rate_limit_error(exc):
                raise
            print(
                f"qa_call_retry attempt={attempt} wait_seconds={retry_seconds} reason={type(exc).__name__}"
            )
            time.sleep(retry_seconds)


def _gemini_generate_json_for_segment(
    record: SegmentRecord,
    *,
    prior_context: str,
    background_text: str | None,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for Gemini QA generation. Install with: pip install google-genai"
        ) from exc

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.")

    text = _read_segment_text(record)
    raw = json.loads(record.raw_path.read_text(encoding="utf-8"))
    time_window = f"{raw.get('start_seconds')}s - {raw.get('end_seconds')}s"
    prompt = (
        "You are creating pathology learning benchmark questions from one segment.\n"
        "Return strict JSON only with these keys:\n"
        "- skip (boolean)\n"
        "- skip_reason (string or null)\n"
        "- question (string or null)\n"
        "- options (array of exactly 4 strings or null)\n"
        "- answer_index (integer 0..3 or null)\n"
        "- explanation (string or null)\n"
        "- evidence_text (string or null)\n\n"
        "Rules:\n"
        "1) If this segment text is too short, ambiguous, or not suitable for a meaningful MCQ, set skip=true.\n"
        "2) If suitable, set skip=false and generate exactly one high-quality MCQ.\n"
        "3) Question must require pathology reasoning grounded in the segment image+text.\n"
        "4) Do not output markdown.\n\n"
        f"Case background: {background_text if background_text else '(none)'}\n"
        f"Segment time window: {time_window}\n"
        f"Prior transcript context (previous segments):\n{prior_context if prior_context else '(none)'}\n\n"
        f"Segment text: {text}\n"
    )

    client = genai.Client(api_key=api_key)

    def _invoke():
        with record.union_path.open("rb") as fh:
            image_bytes = fh.read()
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

    response = _call_with_retry(_invoke, retries=retries, retry_seconds=retry_seconds)
    text_response = getattr(response, "text", None)
    if not text_response:
        raise ValueError(f"Gemini returned empty response text for segment {record.segment_key}")
    parsed = json.loads(text_response)
    if not isinstance(parsed, dict):
        raise ValueError(f"Gemini returned non-object JSON for segment {record.segment_key}")
    return parsed


def _gemini_refine_text_for_segment(
    record: SegmentRecord,
    *,
    prior_context: str,
    background_text: str | None,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for Gemini refinement. Install with: pip install google-genai"
        ) from exc

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.")

    text_raw = _read_segment_text(record)
    raw = json.loads(record.raw_path.read_text(encoding="utf-8"))
    time_window = f"{raw.get('start_seconds')}s - {raw.get('end_seconds')}s"
    prompt = (
        "You are refining pathology teaching narration into concise, formal training text.\n"
        "Return strict JSON only with keys:\n"
        "- skip (boolean)\n"
        "- skip_reason (string or null)\n"
        "- refined_text (string or null)\n"
        "- confidence (number 0..1 or null)\n\n"
        "Rules:\n"
        "1) Use case metadata + prior context + current segment image + current sentence.\n"
        "2) Preserve factual meaning, do not hallucinate unsupported findings.\n"
        "3) Write 1-3 sentences, concise and medically clear.\n"
        "4) If current sentence is too weak/ambiguous to refine reliably, set skip=true.\n"
        "5) Do not output markdown.\n\n"
        f"Case background: {background_text if background_text else '(none)'}\n"
        f"Segment time window: {time_window}\n"
        f"Prior transcript context:\n{prior_context if prior_context else '(none)'}\n\n"
        f"Current segment raw text:\n{text_raw}\n"
    )

    client = genai.Client(api_key=api_key)

    def _invoke():
        with record.union_path.open("rb") as fh:
            image_bytes = fh.read()
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

    response = _call_with_retry(_invoke, retries=retries, retry_seconds=retry_seconds)
    text_response = getattr(response, "text", None)
    if not text_response:
        raise ValueError(f"Gemini returned empty refinement response for segment {record.segment_key}")
    parsed = json.loads(text_response)
    if not isinstance(parsed, dict):
        raise ValueError(f"Gemini returned non-object JSON for refinement segment {record.segment_key}")
    return parsed


def generate_refinement_map(
    records: list[SegmentRecord],
    *,
    background_text: str | None,
    context_window_size: int,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
    max_items: int,
) -> dict[str, dict[str, Any]]:
    segment_text_lookup = build_segment_text_lookup(records)
    refinement_map: dict[str, dict[str, Any]] = {}
    processed = 0
    for idx, record in enumerate(records):
        if max_items > 0 and processed >= max_items:
            break
        processed += 1
        prior_context = prior_context_text(
            records,
            idx,
            segment_text_lookup=segment_text_lookup,
            window_size=context_window_size,
        )
        parsed = _gemini_refine_text_for_segment(
            record,
            prior_context=prior_context,
            background_text=background_text,
            model=model,
            api_key=api_key,
            retries=retries,
            retry_seconds=retry_seconds,
        )
        raw_text = segment_text_lookup.get(record.segment_key, "")
        if parsed.get("skip", False):
            refinement_map[record.segment_key] = {
                "applied": False,
                "skip_reason": parsed.get("skip_reason", "model_skip"),
                "text_refined": None,
                "confidence": parsed.get("confidence"),
            }
            continue
        refined_text = str(parsed.get("refined_text", "")).strip()
        if not refined_text:
            refinement_map[record.segment_key] = {
                "applied": False,
                "skip_reason": "empty_refined_text",
                "text_refined": None,
                "confidence": parsed.get("confidence"),
            }
            continue
        refinement_map[record.segment_key] = {
            "applied": True,
            "skip_reason": None,
            "text_refined": refined_text,
            "text_raw": raw_text,
            "confidence": parsed.get("confidence"),
            "model": model,
        }
        print(f"refined segment={record.segment_key}")
    return refinement_map


def generate_qa_with_gemini(
    records: list[SegmentRecord],
    *,
    background_text: str | None,
    context_window_size: int,
    question_count: int,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    skipped_segments: list[dict[str, Any]] = []
    segment_text_lookup = build_segment_text_lookup(records)

    for idx, record in enumerate(records):
        if len(questions) >= question_count:
            break

        text = _read_segment_text(record)
        if _segment_is_trivially_unusable(text):
            skipped_segments.append(
                {
                    "segment_key": record.segment_key,
                    "reason": "prefilter_short_or_low_context_text",
                }
            )
            continue

        prior_context = prior_context_text(
            records,
            idx,
            segment_text_lookup=segment_text_lookup,
            window_size=context_window_size,
        )
        parsed = _gemini_generate_json_for_segment(
            record,
            prior_context=prior_context,
            background_text=background_text,
            model=model,
            api_key=api_key,
            retries=retries,
            retry_seconds=retry_seconds,
        )

        if parsed.get("skip", False):
            skipped_segments.append(
                {
                    "segment_key": record.segment_key,
                    "reason": parsed.get("skip_reason", "model_skip"),
                }
            )
            continue

        options = parsed.get("options")
        answer_index = parsed.get("answer_index")
        if not isinstance(options, list) or len(options) != 4:
            skipped_segments.append(
                {
                    "segment_key": record.segment_key,
                    "reason": "invalid_options_shape",
                }
            )
            continue
        try:
            answer_index = int(answer_index)
        except (TypeError, ValueError):
            skipped_segments.append(
                {
                    "segment_key": record.segment_key,
                    "reason": "invalid_answer_index",
                }
            )
            continue
        if answer_index < 0 or answer_index > 3:
            skipped_segments.append(
                {
                    "segment_key": record.segment_key,
                    "reason": "answer_index_out_of_range",
                }
            )
            continue

        questions.append(
            {
                "id": len(questions) + 1,
                "segment_key": record.segment_key,
                "image_path": str(record.union_path),
                "question": str(parsed.get("question", "")).strip(),
                "options": [str(opt) for opt in options],
                "answer_index": answer_index,
                "explanation": str(parsed.get("explanation", "")).strip(),
                "evidence_text": str(parsed.get("evidence_text", "")).strip(),
            }
        )
        print(f"qa_generated segment={record.segment_key} total={len(questions)}")

    return {
        "questions": questions,
        "skipped_segments": skipped_segments,
    }


def export_qa_benchmark(
    transcript_json: Path,
    records: list[SegmentRecord],
    output_path: Path,
    *,
    background_text: str | None,
    context_window_size: int,
    provider: str,
    question_count: int,
    model: str,
    api_key: str | None,
    retries: int,
    retry_seconds: int,
) -> Path:
    if provider == "openai":
        transcript_text = load_transcript_text(transcript_json)
        if not transcript_text:
            raise ValueError(f"No transcript text found in {transcript_json}")
        qa_payload = generate_qa_with_openai(
            transcript_text,
            question_count=question_count,
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
    else:
        qa_payload = generate_qa_with_gemini(
            records,
            background_text=background_text,
            context_window_size=context_window_size,
            question_count=question_count,
            model=model,
            api_key=api_key,
            retries=retries,
            retry_seconds=retry_seconds,
        )

    output = {
        "format": "pathlearn.qa.v1",
        "provider": provider,
        "source_transcript": str(transcript_json),
        "model": model,
        "question_count_requested": question_count,
        "questions": qa_payload.get("questions", []),
        "skipped_segments": qa_payload.get("skipped_segments", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    transcript_json = (repo_root / args.transcript_json).resolve()
    case_context_path = (repo_root / args.case_context_json).resolve() if args.case_context_json else None
    case_context = load_case_context(case_context_path) if case_context_path else None

    records = discover_segments(dataset_root)
    if not records:
        raise FileNotFoundError(f"No seg_*_raw.json files found in {dataset_root}")

    print(f"segments={len(records)} dataset_root={dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    background_text, background_info = resolve_background_text(
        case_context=case_context,
        arg_background_text=args.background_text,
        auto_background=args.auto_background,
        records=records,
        output_dir=output_dir,
        transcript_json=transcript_json,
        background_model=args.background_model,
        api_key=args.api_key or os.getenv("GEMINI_API_KEY"),
        retries=args.refine_max_retries,
        retry_seconds=args.refine_retry_seconds,
        grid_size=args.background_grid_size,
        sample_count=args.background_sample_count,
        allow_missing_case_context=args.allow_missing_case_context,
    )
    print(f"background_source={background_info.get('source')} background_present={bool(background_text)}")
    refinement_map: dict[str, dict[str, Any]] | None = None
    if args.refine_text:
        refinement_map = generate_refinement_map(
            records,
            background_text=background_text,
            context_window_size=args.context_window_size,
            model=args.refine_model,
            api_key=args.api_key or os.getenv("GEMINI_API_KEY"),
            retries=args.refine_max_retries,
            retry_seconds=args.refine_retry_seconds,
            max_items=args.refine_max_items,
        )
        refined_count = sum(1 for v in refinement_map.values() if v.get("applied"))
        print(f"refinement_applied={refined_count}")

    eye_attention_map: dict[str, str] | None = None
    if not args.skip_eye_attention:
        eye_attention_map = build_eye_attention_images(
            records,
            output_dir=output_dir,
            repo_root=repo_root,
            attention_tier=args.eye_attention_tier,
            grid_size=args.eye_attention_grid_size,
            sigma=args.eye_attention_sigma,
            quantile=args.eye_attention_quantile,
            show_points=args.eye_attention_show_points,
            render_mode=args.eye_attention_render_mode,
            boundary_color=args.eye_attention_boundary_color,
            boundary_width=args.eye_attention_boundary_width,
        )
        print(
            "eye_attention_generated="
            f"{len(eye_attention_map)} tier={args.eye_attention_tier} "
            f"grid={args.eye_attention_grid_size} mode={args.eye_attention_render_mode}"
        )

    if not args.skip_pdf:
        pdf_path = export_review_pdf(
            records,
            output_dir / "learning_review.pdf",
            case_context=case_context,
            background_text=background_text,
            context_window_size=args.context_window_size,
            refinement_map=refinement_map,
        )
        print(f"pdf={pdf_path}")

    if not args.skip_grounded:
        grounded_path = export_grounded_jsonl(
            records,
            output_dir / "text_image_grounding.jsonl",
            repo_root=repo_root,
            case_context=case_context,
            background_text=background_text,
            background_info=background_info,
            context_window_size=args.context_window_size,
            refinement_map=refinement_map,
            eye_attention_map=eye_attention_map,
        )
        print(f"grounded_jsonl={grounded_path}")
        human_path = export_human_readable_jsonl(
            records,
            output_dir / "human_readable.jsonl",
            repo_root=repo_root,
            case_context=case_context,
            background_text=background_text,
            background_info=background_info,
            context_window_size=args.context_window_size,
            refinement_map=refinement_map,
            eye_attention_map=eye_attention_map,
        )
        print(f"human_readable_jsonl={human_path}")

    if not args.skip_qa:
        qa_path = export_qa_benchmark(
            transcript_json,
            records,
            output_dir / "qa_benchmark.json",
            background_text=background_text,
            context_window_size=args.context_window_size,
            provider=args.qa_provider,
            question_count=args.qa_count,
            model=args.qa_model,
            api_key=args.api_key,
            retries=args.qa_max_retries,
            retry_seconds=args.qa_retry_seconds,
        )
        print(f"qa={qa_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
