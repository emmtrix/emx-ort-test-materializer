"""Validation helpers for replaying extracted ONNX test case artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def dataset_sort_key(path: Path) -> tuple[int, str]:
    """Sort test_data_set directories by numeric suffix."""
    match = re.fullmatch(r"test_data_set_(\d+)", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def tensor_file_sort_key(path: Path) -> tuple[int, str]:
    """Sort input_*.pb and output_*.pb files by numeric suffix."""
    match = re.fullmatch(r"(?:input|output)_(\d+)\.pb", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def load_tensor_proto_array(path: Path) -> np.ndarray:
    """Load a TensorProto from disk and convert it to a numpy array."""
    tensor = onnx.TensorProto()
    tensor.ParseFromString(path.read_bytes())
    return numpy_helper.to_array(tensor)


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Compare two numpy arrays and return a status plus a concise message."""
    if actual.shape != expected.shape:
        return False, f"shape mismatch: actual={actual.shape}, expected={expected.shape}"

    if actual.dtype != expected.dtype:
        return False, f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}"

    if np.issubdtype(actual.dtype, np.inexact):
        if np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
            return True, "allclose"

        return False, "values differ"

    if np.array_equal(actual, expected):
        return True, "exact"

    return False, "values differ"


def resolve_test_case_path(path: Path) -> tuple[Path, list[Path]]:
    """Resolve a user-provided test case path to the test case directory and datasets."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    if resolved.is_dir() and re.fullmatch(r"test_data_set_\d+", resolved.name):
        test_case_dir = resolved.parent
        datasets = [resolved]
    else:
        test_case_dir = resolved
        datasets = sorted(
            [
                entry
                for entry in test_case_dir.iterdir()
                if entry.is_dir() and re.fullmatch(r"test_data_set_\d+", entry.name)
            ],
            key=dataset_sort_key,
        )

    if not test_case_dir.is_dir():
        raise ValueError(f"Expected a test case directory or test_data_set directory: {resolved}")

    if not datasets:
        raise ValueError(f"No test_data_set_* directories found under: {test_case_dir}")

    return test_case_dir, datasets


def validate_dataset_result(
    session: ort.InferenceSession,
    dataset_dir: Path,
    *,
    atol: float,
    rtol: float,
) -> str:
    """Return OK or the first failure message for one dataset directory."""
    input_paths = sorted(dataset_dir.glob("input_*.pb"), key=tensor_file_sort_key)
    output_paths = sorted(dataset_dir.glob("output_*.pb"), key=tensor_file_sort_key)

    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    if len(input_paths) != len(session_inputs):
        return f"{dataset_dir.name}: input count mismatch: files={len(input_paths)}, model_inputs={len(session_inputs)}"

    if len(output_paths) != len(session_outputs):
        return f"{dataset_dir.name}: output count mismatch: files={len(output_paths)}, model_outputs={len(session_outputs)}"

    feeds: dict[str, np.ndarray] = {}
    for input_info, input_path in zip(session_inputs, input_paths, strict=True):
        feeds[input_info.name] = load_tensor_proto_array(input_path)

    computed_outputs = session.run(None, feeds)

    for index, (output_info, output_path, actual_output) in enumerate(
        zip(session_outputs, output_paths, computed_outputs, strict=True)
    ):
        expected_output = load_tensor_proto_array(output_path)
        matches, detail = compare_arrays(
            np.asarray(actual_output),
            np.asarray(expected_output),
            atol=atol,
            rtol=rtol,
        )
        if not matches:
            return f"{dataset_dir.name} output_{index} ({output_info.name}): FAIL - {detail}"

    return "OK"


def validate_test_case_result(
    test_case_path: Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> str:
    """Return OK or the first failure message for one extracted test case."""
    try:
        test_case_dir, datasets = resolve_test_case_path(test_case_path)
        model_path = test_case_dir / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        session = ort.InferenceSession(
            model_path.as_posix(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        for dataset_dir in datasets:
            result = validate_dataset_result(
                session,
                dataset_dir,
                atol=atol,
                rtol=rtol,
            )
            if result != "OK":
                return result
        return "OK"
    except (FileNotFoundError, ValueError, Exception) as error:
        return str(error)
