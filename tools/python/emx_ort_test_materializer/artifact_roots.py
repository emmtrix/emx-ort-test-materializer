"""Helpers for locating tracked artifact dataset roots.

This repository may store ORT artifacts under more than one tracked root.
Positive cases live under ``onnxruntime/`` and expected-failure cases may live
under ``onnxruntime-negative/``. These helpers keep discovery logic in one
place so reporting code can load one or both roots consistently.
"""

from __future__ import annotations

from pathlib import Path

PRIMARY_ARTIFACT_ROOT_NAME = "onnxruntime"
NEGATIVE_ARTIFACT_ROOT_NAME = "onnxruntime-negative"
TRACKED_ARTIFACT_ROOT_NAMES = (
    PRIMARY_ARTIFACT_ROOT_NAME,
    NEGATIVE_ARTIFACT_ROOT_NAME,
)


def discover_artifact_dataset_roots(artifacts_root: Path) -> tuple[Path, ...]:
    """Return tracked dataset roots below ``artifacts_root`` or the root itself."""
    discovered = tuple(
        root
        for root in (artifacts_root / name for name in TRACKED_ARTIFACT_ROOT_NAMES)
        if root.is_dir()
    )
    if discovered:
        return discovered
    return (artifacts_root,)


def artifact_case_path_base(artifacts_root: Path, dataset_roots: tuple[Path, ...]) -> Path:
    """Return the base used when rendering case paths.

    With a single discovered dataset root, keep paths relative to that root so
    existing reports stay stable. When multiple roots are present, switch to the
    common ``artifacts/`` parent so case paths include ``onnxruntime`` versus
    ``onnxruntime-negative``.
    """

    if len(dataset_roots) == 1:
        return dataset_roots[0]
    return artifacts_root
