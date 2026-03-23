# emx-ort-test-materializer

A lightweight internal utility to materialize ONNX and TensorProto `.pb` artifacts
from the official [ONNX Runtime](https://github.com/microsoft/onnxruntime) test suite —
covering **both Python and C++ tests**.

---

## Motivation

ONNX Runtime ships a large test suite spanning Python scripts and C++ binaries.
Both kinds of tests exercise many operators and contrib-ops by constructing models
and feeding inputs either programmatically (Python) or through pre-existing test-data
files (C++).  There is currently no easy way to obtain the corresponding `.onnx` model
files and serialized `input_*.pb` / `output_*.pb` tensors as standalone artifacts
outside of a full ORT build.

This tool extracts those artifacts by:

- **Python tests** – instrumenting the Python test modules to intercept
  `InferenceSession` calls and capture model bytes and numpy tensors at runtime.
- **C++ tests** – extracting reusable JSON metadata from ORT `OpTester`-based
  C++ tests and locating any pre-existing test-data files already present in
  the ORT source tree.

The resulting artifacts are committed to this repository so that downstream consumers
(e.g. emmtrix compiler test-suites) can reference them without requiring a running ORT
installation or a full build.

---

## Scope

- Read test sources and data from the bundled ONNX Runtime submodule.
- **Python tests**: instrument `InferenceSession` / helper calls to capture model and
  input/output tensors.
- **C++ tests**: discover and collect test-data files (`.onnx`, `.pb`) referenced by
  C++ test sources, and extract JSON rules from `OpTester`-based Microsoft/contrib-op
  tests for later Python-side materialization.
- Write all artifacts to `artifacts/` in a layout that mirrors the ORT test directory tree.
- Commit the resulting artifacts to this repository for reproducible consumption.

---

## Non-Goals

- This is **not** a distributable Python package.
- It does **not** run the full ORT test-suite.
- It does **not** modify ONNX Runtime source code.
- It does **not** provide CI or automated publishing.
- It does **not** generate final `.onnx` or `.pb` artifacts from C++ tests yet.

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
      testdata/
        <op_or_suite_name>/
          model.onnx
          test_data_set_0/
            input_0.pb
            output_0.pb
      providers/
        <provider_test_name>/
          model.onnx
          test_data_set_0/
            input_0.pb
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
│   └── extract_test_artifacts.py     # builds and runs the C++ extractor
├── cpp/
│   └── ort_cpp_test_extractor.cpp    # standalone C++ source extractor
├── metadata/
│   └── ort_cpp_test_rules.json       # checked-in extracted C++ test metadata
├── src/
│   └── emx_ort_test_materializer/
│       ├── __init__.py
│       ├── extractor.py              # future extraction logic (Python + C++)
│       └── writer.py                 # future serialization logic
├── artifacts/
│   └── README.md                     # layout documentation
└── onnxruntime-org/
    └── onnxruntime/                  # git submodule
```

---

## Planned Next Steps

1. **Python test instrumentation** – wrap `onnxruntime.InferenceSession` and
   related helpers in the contrib-ops test files to intercept model bytes and
   numpy input arrays before execution.
2. **C++ metadata extraction** – scan ORT `OpTester`-based C++ tests and emit JSON
   rules containing operator metadata, attributes, and input tensors where they can
   be resolved statically.
3. **Serialize Python-test models** – write captured ONNX `ModelProto` objects to
   `model.onnx` using `model.SerializeToString()`.
4. **Serialize Python-test tensors** – convert numpy arrays to `onnx.TensorProto`
   with `numpy_helper.from_array()` and write them as `.pb` files.
5. **Map to artifact paths** – derive the output path from the test file name and
   test-case name, following the artifact layout described above.
6. **Validate** – spot-check a handful of produced artifacts with
   `onnxruntime.InferenceSession` to confirm round-trip correctness.

---

## Current C++ Extractor

The repository now includes a standalone C++ extractor at
`cpp/ort_cpp_test_extractor.cpp` and a thin Python wrapper at
`scripts/extract_test_artifacts.py`.

Current behavior:

- Scans ORT C++ test sources for `OpTester`-based tests.
- Extracts JSON rules for Microsoft-domain (`kMSDomain`) tests by default.
- Captures operator name, opset, domain, attributes, and input/output tensor
  definitions when they can be resolved statically.
- Marks each extracted record as `complete`, `partial`, or `metadata_only`
  depending on how much tensor data could be resolved.

Example:

```bash
python scripts/extract_test_artifacts.py \
  --cpp-source onnxruntime-org/onnxruntime/test/contrib_ops
```

The generated JSON is intended to be consumed by later Python steps that build
`ModelProto` objects and materialize optional `.pb` inputs/outputs. By default,
the checked-in output is written to `metadata/ort_cpp_test_rules.json`, while
temporary build products stay under the ignored `build/` directory.

---

## License

MIT – see [LICENSE](LICENSE).
