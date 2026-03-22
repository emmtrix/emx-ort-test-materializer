"""
extractor.py
------------
Future home of the test-instrumentation and artifact-extraction logic.

Planned responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~
- Discover relevant test modules under the ONNX Runtime submodule at
  ``onnxruntime-org/onnxruntime/onnxruntime/test/python/contrib_ops/``.
- Monkey-patch ``onnxruntime.InferenceSession`` (and related helpers) so that
  model bytes and numpy input/output arrays are intercepted before and after
  each test invocation.
- Yield ``(test_file, test_case, model_bytes, inputs, outputs)`` tuples that
  the writer module can then serialize to disk.

This module is a **placeholder only**.  No extraction logic is implemented yet.
See ``AGENTS.md`` for the full implementation roadmap.
"""

from __future__ import annotations


def discover_test_files(base_path: str) -> list[str]:
    """Placeholder – will return paths to ORT contrib-ops test files.

    Parameters
    ----------
    base_path:
        Root of the ONNX Runtime source tree, e.g.
        ``"onnxruntime-org/onnxruntime"``.

    Returns
    -------
    list[str]
        Absolute paths to Python test files.  Always empty until implemented.
    """
    # TODO: implement discovery logic
    return []


def extract_artifacts(test_file: str) -> list[dict]:
    """Placeholder – will instrument a single test file and capture artifacts.

    Parameters
    ----------
    test_file:
        Absolute path to the Python test module to instrument.

    Returns
    -------
    list[dict]
        List of artifact records, each containing ``test_case``,
        ``model_bytes``, ``inputs``, and ``outputs`` keys.
        Always empty until implemented.
    """
    # TODO: implement instrumentation and extraction logic
    return []
