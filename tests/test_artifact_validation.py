"""Pytest coverage for extracted ORT artifact replay expectations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import validate_test_case_result


EXPECTATIONS_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"


def load_expectations() -> list[dict[str, str]]:
    """Load expected validation outcomes for all checked-in artifact test cases."""
    payload = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    cases = payload["test_cases"]
    return sorted(cases, key=lambda case: case["path"])


EXPECTED_CASES = load_expectations()


@pytest.mark.parametrize(
    ("relative_path", "expected_result"),
    [(case["path"], case["expected_result"]) for case in EXPECTED_CASES],
    ids=[case["path"] for case in EXPECTED_CASES],
)
def test_artifact_validation_result(relative_path: str, expected_result: str) -> None:
    """Replay one artifact directory and compare its result string against the stored expectation."""
    actual_result = validate_test_case_result(REPO_ROOT / relative_path)
    assert actual_result == expected_result
