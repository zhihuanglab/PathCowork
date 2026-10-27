#!/usr/bin/env python3
"""Build a unified review manifest for human/machine/eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewer manifest JSON.")
    parser.add_argument(
        "--dataset-root",
        default="history/case_003",
        help="Processed segment folder containing seg_00xx_* files",
    )
    parser.add_argument(
        "--human-jsonl",
        default="final_outputs/example_data/human_readable.jsonl",
        help="Human-readable JSONL with refined text (optional if missing)",
    )
    parser.add_argument(
        "--grounded-jsonl",
        default="final_outputs/example_data/text_image_grounding.jsonl",
        help="Machine-friendly grounded dataset JSONL (optional if missing)",
    )
    parser.add_argument(
        "--qa-json",
        default="final_outputs/example_data/qa_benchmark.json",
        help="Evaluation/benchmark QA JSON (optional if missing)",
    )
    parser.add_argument(
        "--output",
        default="reviewers/data/review_manifest.json",
        help="Output manifest path",
    )
    return parser.parse_args()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def build_human_items_from_jsonl(human_jsonl: Path, repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not human_jsonl.exists():
        return items
    with human_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seg_key = str(row.get("segment_key"))
            text_raw = row.get("text_raw", "")
            text_refined = row.get("text_refined")
            context_text = row.get("text_prior_context", "")
            merged = (
                f"RAW:\n{text_raw}\n\n"
                f"REFINED:\n{text_refined if text_refined else '(not refined)'}\n\n"
                f"PRIOR CONTEXT:\n{context_text if context_text else '(none)'}"
            )
            items.append(
                {
                    "id": f"human_{seg_key}",
                    "type": "human_readable",
                    "title": f"Segment {seg_key}",
                    "text": merged,
                    "images": [
                        row.get("image_union"),
                        row.get("image_eye_attention"),
                    ],
                    "meta": {
                        "segment_key": seg_key,
                        "start_seconds": row.get("start_seconds"),
                        "end_seconds": row.get("end_seconds"),
                        "background_text": row.get("background_text"),
                        "background_info": row.get("background_info"),
                        "case_context": row.get("case_context"),
                        "refinement": row.get("text_refinement"),
                    },
                }
            )
    return items


def build_human_items(dataset_root: Path, repo_root: Path, human_jsonl: Path) -> list[dict[str, Any]]:
    from_jsonl = build_human_items_from_jsonl(human_jsonl, repo_root)
    if from_jsonl:
        return from_jsonl
    items: list[dict[str, Any]] = []
    for raw_path in sorted(dataset_root.glob("seg_*_raw.json")):
        seg_key = raw_path.name.split("_")[1]
        text_path = dataset_root / f"seg_{seg_key}_text.txt"
        union_path = dataset_root / f"seg_{seg_key}_union.png"
        if not union_path.exists():
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else str(raw.get("text", "")).strip()
        items.append(
            {
                "id": f"human_{seg_key}",
                "type": "human_readable",
                "title": f"Segment {seg_key}",
                "text": text,
                "images": [_repo_relative(union_path, repo_root)],
                "meta": {
                    "segment_key": seg_key,
                    "start_seconds": raw.get("start_seconds"),
                    "end_seconds": raw.get("end_seconds"),
                    "refinement": {"applied": False, "reason": "no_human_jsonl"},
                },
            }
        )
    return items


def _build_machine_items_from_raw(dataset_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_path in sorted(dataset_root.glob("seg_*_raw.json")):
        seg_key = raw_path.name.split("_")[1]
        union_path = dataset_root / f"seg_{seg_key}_union.png"
        eye_path = dataset_root / f"seg_{seg_key}_eye.png"
        mouse_path = dataset_root / f"seg_{seg_key}_mouse.png"
        text_path = dataset_root / f"seg_{seg_key}_text.txt"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else str(raw.get("text", "")).strip()
        items.append(
            {
                "id": f"machine_{seg_key}",
                "type": "machine_friendly",
                "title": f"Grounded Item {seg_key}",
                "text": text,
                "images": [
                    _repo_relative(union_path, repo_root) if union_path.exists() else None,
                    _repo_relative(eye_path, repo_root) if eye_path.exists() else None,
                    _repo_relative(mouse_path, repo_root) if mouse_path.exists() else None,
                ],
                "meta": {
                    "segment_key": seg_key,
                    "start_seconds": raw.get("start_seconds"),
                    "end_seconds": raw.get("end_seconds"),
                    "has_boxes_final": bool(raw.get("boxes_final")),
                    "eye_points": len(raw.get("eye") or []),
                    "mouse_points": len(raw.get("mouse") or []),
                    "source": "raw_fallback",
                },
            }
        )
    return items


def build_machine_items(grounded_jsonl: Path, dataset_root: Path, repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not grounded_jsonl.exists():
        return _build_machine_items_from_raw(dataset_root, repo_root)
    with grounded_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seg_key = str(row.get("segment_key"))
            images = row.get("images") or {}
            items.append(
                {
                    "id": f"machine_{seg_key}",
                    "type": "machine_friendly",
                    "title": f"Grounded Item {seg_key}",
                    "text": (
                        f"RAW:\n{row.get('text_raw','')}\n\n"
                        f"REFINED:\n{row.get('text_refined') or '(not refined)'}\n\n"
                        f"PRIOR CONTEXT:\n{row.get('text_prior_context') or '(none)'}"
                    ),
                    "images": [
                        images.get("union"),
                        images.get("eye"),
                        images.get("mouse"),
                        images.get("eye_attention"),
                    ],
                    "meta": {
                        "segment_key": seg_key,
                        "start_seconds": row.get("start_seconds"),
                        "end_seconds": row.get("end_seconds"),
                        "has_boxes_final": bool((row.get("grounding") or {}).get("boxes_final")),
                        "eye_points": len((row.get("grounding") or {}).get("eye_points") or []),
                        "mouse_points": len((row.get("grounding") or {}).get("mouse_points") or []),
                        "background_text": row.get("background_text"),
                        "refinement": row.get("text_refinement"),
                        "has_eye_attention": bool(images.get("eye_attention")),
                    },
                }
            )
    return items if items else _build_machine_items_from_raw(dataset_root, repo_root)


def build_eval_items(qa_json: Path, repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not qa_json.exists():
        return items
    payload = json.loads(qa_json.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    for q in questions:
        qid = q.get("id")
        image_path = q.get("image_path")
        image_rel = _repo_relative(Path(image_path), repo_root) if image_path else None
        options = q.get("options") or []
        text = q.get("question", "")
        if options:
            options_text = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options))
            text = f"{text}\n\nOptions:\n{options_text}"
        items.append(
            {
                "id": f"eval_{qid}",
                "type": "evaluation",
                "title": f"QA {qid}",
                "text": text,
                "images": [image_rel] if image_rel else [],
                "meta": {
                    "segment_key": q.get("segment_key"),
                    "answer_index": q.get("answer_index"),
                    "explanation": q.get("explanation"),
                    "evidence_text": q.get("evidence_text"),
                },
            }
        )
    return items


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    dataset_root = (repo_root / args.dataset_root).resolve()
    human_jsonl = (repo_root / args.human_jsonl).resolve()
    grounded_jsonl = (repo_root / args.grounded_jsonl).resolve()
    qa_json = (repo_root / args.qa_json).resolve()
    output_path = (repo_root / args.output).resolve()

    manifest = {
        "format": "pathlearn.reviewer.v1",
        "counts": {},
        "items": [],
    }

    human = build_human_items(dataset_root, repo_root, human_jsonl)
    machine = build_machine_items(grounded_jsonl, dataset_root, repo_root)
    evaluation = build_eval_items(qa_json, repo_root)

    items = human + machine + evaluation
    manifest["counts"] = {
        "human_readable": len(human),
        "machine_friendly": len(machine),
        "evaluation": len(evaluation),
        "total": len(items),
    }
    manifest["items"] = items

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"manifest={output_path}")
    print(f"counts={manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
