# Artifacts

This directory is the primary payload of the repository.

It stores checked-in ONNX models, TensorProto `.pb` files, and validation
metadata derived from ONNX Runtime tests. Consumers should start here, not in
the maintenance tooling under `tools/`.

## Contents

- `onnxruntime/`: tracked artifact dataset mirroring ONNX Runtime test sources.
- `MANIFEST.json`: dataset-level metadata, including the pinned source version.
- `VALIDATION_ERRORS.md`: generated summary of non-OK validation outcomes and ignored cases.

All files in this directory are committed so downstream tooling can consume
them without rebuilding ONNX Runtime locally.

## Layout

```text
artifacts/
  onnxruntime/
    test/
      python/
        contrib_ops/
          <test_file>/
            <test_case>/
              model.onnx
              validation.json
              test_data_set_0/
                input_0.pb
                input_1.pb
                output_0.pb
      contrib_ops/
        <test_file>/
          <test_name>_run<n>/
            model.onnx
            validation.json
            test_data_set_0/
              input_0.pb
              output_0.pb
      testdata/
        <name>/
          ...
      providers/
        <name>/
          ...
```

## Source-to-Artifact Mapping

| ORT source path | Artifact sub-path | Extraction method |
| --- | --- | --- |
| `onnxruntime/test/python/contrib_ops/<file>.py` | `onnxruntime/test/python/contrib_ops/<file>/<test_case>/` | Python runtime instrumentation |
| `onnxruntime/test/python/<file>.py` | `onnxruntime/test/python/<file>/<test_case>/` | Python runtime instrumentation |
| `onnxruntime/test/contrib_ops/<file>.cc` | `onnxruntime/test/contrib_ops/<file>/<test_name>_run<n>/` | Runtime `OpTester` wrapper |
| `onnxruntime/test/testdata/<name>/` | `onnxruntime/test/testdata/<name>/` | Static copy from ORT checkout |
| `onnxruntime/test/providers/<name>/` | `onnxruntime/test/providers/<name>/` | Static copy from ORT checkout |

## File Formats

- `model.onnx`: binary ONNX `ModelProto`.
- `input_*.pb` and `output_*.pb`: binary `TensorProto` for dense tensors, or
  `SparseTensorProto` for sparse inputs.
- `validation.json`: replay metadata captured from the originating ORT test.

## Maintainer Context

The artifact refresh and validation commands live in `tools/scripts/` and are
documented in [`../DEVELOPMENT.md`](../DEVELOPMENT.md). Those scripts are
repository maintenance infrastructure, not the main product.
