"""Checks for tracked dataset manifest metadata."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "artifacts" / "MANIFEST.json"


def test_manifest_lists_split_artifact_roots() -> None:
    """Keep manifest metadata aligned with the tracked artifact layout."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert payload["artifact_root"] == "artifacts/onnxruntime"
    assert payload["artifact_roots"] == [
        "artifacts/onnxruntime",
        "artifacts/onnxruntime-negative",
    ]
