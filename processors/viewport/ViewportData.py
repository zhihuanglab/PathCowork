#!/usr/bin/env python3
"""Viewport data reader class for PathLearn coordinates JSONL files."""

from __future__ import annotations

from pathlib import Path

from processors.modality_readers import ViewportReader


class ViewportData(ViewportReader):
    """Named wrapper for viewport JSONL reader and visualization."""

    def __init__(self, path: str | Path):
        super().__init__(path)

