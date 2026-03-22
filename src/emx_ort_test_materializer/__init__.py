"""
emx_ort_test_materializer
--------------------------
Lightweight internal utility for materializing ONNX and TensorProto ``.pb``
artifacts from ONNX Runtime Python tests.

This package is **not** intended for distribution.  It is a collection of
helper modules used by the scripts under ``scripts/``.

Modules
~~~~~~~
- :mod:`extractor` – future logic for discovering and instrumenting ORT test
  files to intercept model bytes and numpy tensors.
- :mod:`writer` – future logic for serializing ``ModelProto`` objects and
  numpy arrays to disk in the artifact layout.
"""
