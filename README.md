# emx-ort-test-materializer

A lightweight internal utility to materialize ONNX and TensorProto `.pb` artifacts
from the official [ONNX Runtime](https://github.com/microsoft/onnxruntime) Python tests.

---

## Motivation

ONNX Runtime ships a large test suite under `onnxruntime/test/python/`.
Those tests exercise many operators and contrib-ops through Python APIs that
construct models and feed inputs on the fly.  There is currently no easy way to
obtain the corresponding `.onnx` model files and serialized `input_*.pb` /
`output_*.pb` tensors outside of a full ORT build.

This tool instruments the relevant test files, intercepts the model and tensor
objects at runtime, and serializes them to disk in a well-known layout so that
downstream consumers (e.g. compiler test-suites) can reference them without
requiring a running ORT installation.

---

## Scope

- Read Python test files from the bundled ONNX Runtime submodule.
- Instrument `InferenceSession` / helper calls to capture model and input/output tensors.
- Write `.onnx` and `.pb` files to `artifacts/` in a layout that mirrors the ORT
  test directory tree.
- Commit the resulting artifacts to this repository for reproducible consumption.

---

## Non-Goals

- This is **not** a distributable Python package.
- It does **not** run the full ORT test-suite or validate numerical results.
- It does **not** modify ONNX Runtime source code.
- It does **not** provide CI or automated publishing.

---

## Artifact Layout

Extracted artifacts are stored under `artifacts/` and mirror the ONNX Runtime
source tree:

```
artifacts/
  onnxruntime/
    test/
      python/
        contrib_ops/
          <test_file>/
            <test_case>/
              model.onnx
              test_data_set_0/
                input_0.pb
                input_1.pb   # if multiple inputs
                output_0.pb  # expected output (optional)
```

See [`artifacts/README.md`](artifacts/README.md) for a detailed description of
the layout and naming conventions.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── AGENTS.md
├── .github/
│   └── copilot-instructions.md
├── scripts/
│   └── extract_from_python_tests.py   # entry point (placeholder)
├── src/
│   └── emx_ort_test_materializer/
│       ├── __init__.py
│       ├── extractor.py               # future extraction logic
│       └── writer.py                  # future serialization logic
├── artifacts/
│   └── README.md                      # layout documentation
└── onnxruntime-org/
    └── onnxruntime/                   # git submodule
```

---

## Planned Next Steps

1. **Instrument Python tests** – wrap `onnxruntime.InferenceSession` and
   related helpers in the contrib-ops test files to intercept model bytes and
   numpy input arrays before execution.
2. **Serialize models** – write captured ONNX `ModelProto` objects to
   `model.onnx` using `model.SerializeToString()`.
3. **Serialize tensors** – convert numpy arrays to `onnx.TensorProto` with
   `numpy_helper.from_array()` and write them as `.pb` files.
4. **Map to artifact paths** – derive the output path from the test file name
   and test-case name, following the artifact layout described above.
5. **Validate** – spot-check a handful of produced artifacts with
   `onnxruntime.InferenceSession` to confirm round-trip correctness.

---

## License

MIT – see [LICENSE](LICENSE).
