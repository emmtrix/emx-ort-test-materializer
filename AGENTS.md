# AGENTS.md – Guidance for AI Coding Agents

This file provides project-specific rules and guidance for AI coding agents
(GitHub Copilot, Claude, etc.) working in this repository.

---

## Project Purpose

`emx-ort-test-materializer` is a lightweight internal Python utility that will
instrument ONNX Runtime Python tests and serialize the ONNX models and
TensorProto inputs/outputs they exercise to disk as committed artifacts.

The artifacts are then consumed by downstream tooling (e.g. emmtrix compiler
test-suites) without requiring a full ONNX Runtime build.

---

## Current Phase: Scaffolding Only

The repository is currently in the **scaffolding phase**.  No extraction logic
has been implemented yet.  The primary goal right now is to establish a clean,
extensible project structure.

---

## Constraints

- **Do not execute ONNX Runtime tests** during this phase.
- **Do not generate any artifact files** (`.onnx`, `.pb`) yet.
- Keep all Python files as **placeholders with docstrings** only.
- Do **not** add `pyproject.toml` or packaging configuration – this is a
  non-packaged utility.
- Do **not** add CI workflows, test runners, or complex tooling.
- All code, comments, and documentation must be in **English**.

---

## Coding Conventions

- Python 3.10+.
- Use type hints where they aid clarity.
- Keep modules small and focused on a single responsibility.
- Prefer explicit over implicit; avoid premature abstraction.
- No third-party dependencies beyond those listed in `requirements.txt`.
- Do not modify any file under `onnxruntime-org/onnxruntime/` (submodule).

---

## Rules

1. No execution of ORT tests yet.
2. No generation of artifact files yet.
3. Keep logic minimal and explicit.
4. New modules must include a module-level docstring describing their future
   responsibility before any implementation is added.
5. The `artifacts/` directory must remain tracked by git (never add it to
   `.gitignore`).

---

## Artifact Layout

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
                output_0.pb
```

The layout mirrors the path structure of the ONNX Runtime source tree so that
each artifact can be traced back to its originating test.

---

## Guidance for Future Implementation Steps

### Step 1 – Instrument Python Tests

- Identify which test files under
  `onnxruntime-org/onnxruntime/onnxruntime/test/python/contrib_ops/` create
  `InferenceSession` objects.
- Wrap or monkey-patch `onnxruntime.InferenceSession.__init__` and `.run()` to
  intercept model bytes and numpy input arrays.

### Step 2 – Serialize Models

- Deserialize the intercepted bytes with `onnx.load_from_string()`.
- Write the `ModelProto` back to disk as `model.onnx` under the appropriate
  artifact path.

### Step 3 – Serialize Tensors

- Convert each numpy input/output array to `onnx.TensorProto` using
  `onnx.numpy_helper.from_array()`.
- Write each proto as a binary `.pb` file (`input_0.pb`, `output_0.pb`, …).

### Step 4 – Path Mapping

- Derive the output directory from the test module name and the test-case name.
- Follow the artifact layout documented above.

### Step 5 – Validation

- After extraction, load each `model.onnx` and `input_*.pb` with
  `onnxruntime.InferenceSession` and `onnx.numpy_helper.to_array()`.
- Compare results against the serialized `output_*.pb` files.
