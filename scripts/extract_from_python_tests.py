#!/usr/bin/env python3
"""
scripts/extract_from_python_tests.py
-------------------------------------
Entry point for materializing ONNX and TensorProto artifacts from ONNX
Runtime Python tests.

Intended future functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Discover test modules under
   ``onnxruntime-org/onnxruntime/onnxruntime/test/python/contrib_ops/``.
2. Instrument ``onnxruntime.InferenceSession`` to intercept model bytes and
   numpy input tensors before each test runs.
3. Serialize intercepted ``ModelProto`` objects to
   ``artifacts/onnxruntime/test/python/contrib_ops/<test_file>/<test_case>/model.onnx``.
4. Serialize intercepted numpy arrays to ``input_0.pb``, ``output_0.pb``, …
   inside a ``test_data_set_0/`` sub-directory next to the model file.
5. Report a summary of produced artifacts.

This script is a **placeholder only**.  Do not add extraction logic here until
the instrumentation approach is agreed upon.  See ``AGENTS.md`` for the full
implementation roadmap.

Usage (future)
~~~~~~~~~~~~~~
    python scripts/extract_from_python_tests.py
"""


def main() -> None:
    """Placeholder entry point.  No logic is executed yet."""
    print("emx-ort-test-materializer: extraction not implemented yet.")
    print("See AGENTS.md for the planned implementation steps.")


if __name__ == "__main__":
    main()
