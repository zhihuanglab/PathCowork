#!/usr/bin/env python3
"""Mouse data reader class for PathLearn mouse JSONL files."""

from __future__ import annotations

from pathlib import Path

from processors.modality_readers import MouseReader


class MouseData(MouseReader):
    """Named wrapper for mouse JSONL reader and visualization."""

    def __init__(self, path: str | Path):
        super().__init__(path)

