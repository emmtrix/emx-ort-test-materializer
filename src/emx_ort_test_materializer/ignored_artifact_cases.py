"""Load configured artifact cases that generation should skip.

This module owns the tracked repository configuration for artifact test cases
that are intentionally ignored during ONNX/TensorProto generation and related
validation reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IGNORED_CASES_PATH = REPO_ROOT / "artifact_generation_ignored_cases.json"


@dataclass(frozen=True)
class IgnoredArtifactCase:
    """Describe one generated artifact case that should be skipped."""

    path: str
    reason: str


def load_ignored_artifact_cases(
    path: Path = DEFAULT_IGNORED_CASES_PATH,
) -> tuple[IgnoredArtifactCase, ...]:
    """Load ignored artifact cases from one tracked JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in payload.get("ignored_cases", []):
        case_path = str(entry["path"]).strip().strip("/")
        reason = str(entry["reason"]).strip()
        if not case_path:
            raise ValueError(f"Ignored artifact case path must not be empty in {path}")
        if not reason:
            raise ValueError(f"Ignored artifact case reason must not be empty in {path}")
        cases.append(IgnoredArtifactCase(path=case_path, reason=reason))
    return tuple(cases)


def ignored_case_reasons_by_path(
    ignored_cases: tuple[IgnoredArtifactCase, ...],
) -> dict[str, str]:
    """Return a stable mapping from artifact-relative path to ignore reason."""
    return {case.path: case.reason for case in ignored_cases}


def artifact_case_display_path(path: str) -> str:
    """Return the display path used in repository documentation."""
    normalized = path.strip().strip("/")
    return f"artifacts/{normalized}"
