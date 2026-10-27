#!/usr/bin/env python3
"""Route PathLearn JSONL events into modality-specific JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_LEVEL_TO_OUTPUT = {
    "eye_movement": "processors/gaze/example_data/{stem}_eye_movement.jsonl",
    "mouse": "processors/mouse/example_data/{stem}_mouse.jsonl",
    "coordinates": "processors/viewport/example_data/{stem}_coordinates.jsonl",
    "open_image": "viewer/example_data/{stem}_viewer_events.jsonl",
    "close_image": "viewer/example_data/{stem}_viewer_events.jsonl",
    "wsi_info": "viewer/example_data/{stem}_viewer_events.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one PathLearn JSONL file into modality-specific JSONL files."
    )
    parser.add_argument("--input", required=True, help="Path to source JSONL log")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output prefix name (e.g., brain). Defaults to input stem.",
    )
    parser.add_argument(
        "--unknown-output",
        default="viewer/example_data/{stem}_other_events.jsonl",
        help="Output path template for unknown levels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    stem = args.name if args.name else src.stem

    writers: dict[str, object] = {}
    level_counts: Counter[str] = Counter()
    bad_lines = 0

    def writer_for(level: str):
        template = DEFAULT_LEVEL_TO_OUTPUT.get(level, args.unknown_output)
        path = Path(template.format(stem=stem))
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path) not in writers:
            writers[str(path)] = path.open("w", encoding="utf-8")
        return writers[str(path)], path

    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            level = obj.get("level", "<missing>")
            fout, _ = writer_for(level)
            fout.write(json.dumps(obj, ensure_ascii=True) + "\n")
            level_counts[level] += 1

    for handle in writers.values():
        handle.close()

    print(f"input={src}")
    for level, count in sorted(level_counts.items()):
        template = DEFAULT_LEVEL_TO_OUTPUT.get(level, args.unknown_output)
        out_path = template.format(stem=stem)
        print(f"level={level} count={count} output={out_path}")
    print(f"bad_lines={bad_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
