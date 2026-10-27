#!/usr/bin/env python3
"""Build review manifest and launch local reviewer server."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import subprocess
import threading
import webbrowser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PathLearn reviewer in browser.")
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP port")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build manifest, do not run server",
    )
    return parser.parse_args()


def build_manifest(repo_root: Path) -> None:
    cmd = ["python3", "reviewers/build_review_manifest.py"]
    subprocess.run(cmd, cwd=repo_root, check=True)


def serve(repo_root: Path, port: int) -> None:
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"reviewer_url=http://127.0.0.1:{port}/reviewers/web/index.html")
        thread = threading.Thread(
            target=lambda: webbrowser.open(f"http://127.0.0.1:{port}/reviewers/web/index.html"),
            daemon=True,
        )
        thread.start()
        httpd.serve_forever()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    build_manifest(repo_root)
    if args.build_only:
        return 0
    serve(repo_root, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

