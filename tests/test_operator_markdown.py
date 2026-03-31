from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.operator_markdown import (
    DOCS_REGEN_COMMAND,
    load_single_operator_cases,
    render_operator_markdown,
    summarize_test_case_paths,
)


REPORT_PATH = REPO_ROOT / "artifacts" / "OPERATORS.md"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


def make_single_operator_model(op_type: str, *, domain: str = "") -> onnx.ModelProto:
    """Create a minimal one-operator model."""
    input_value = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    output_value = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node(op_type, ["X"], ["Y"], domain=domain)],
        "graph",
        [input_value],
        [output_value],
    )
    return helper.make_model(graph)


def make_multi_operator_model() -> onnx.ModelProto:
    """Create a minimal multi-operator model for validation tests."""
    input_value = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    output_value = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["X"], ["mid"]),
            helper.make_node("Identity", ["mid"], ["Y"]),
        ],
        "graph",
        [input_value],
        [output_value],
    )
    return helper.make_model(graph)


def test_load_single_operator_cases_requires_exactly_one_operator(tmp_path: Path) -> None:
    case_dir = tmp_path / "test" / "suite" / "Case_run0"
    case_dir.mkdir(parents=True)
    onnx.save(make_multi_operator_model(), case_dir / "model.onnx")

    with pytest.raises(ValueError, match="Expected exactly one operator"):
        load_single_operator_cases(tmp_path)


def test_summarize_test_case_paths_groups_runs() -> None:
    summarized = summarize_test_case_paths(
        [
            "onnxruntime/test/contrib_ops/demo/TestAlpha_run0",
            "onnxruntime/test/contrib_ops/demo/TestAlpha_run1",
            "onnxruntime/test/contrib_ops/demo/TestAlpha_run2",
            "onnxruntime/test/contrib_ops/demo/TestBeta_run4",
            "onnxruntime/test/contrib_ops/demo/TestBeta_run5",
            "onnxruntime/test/contrib_ops/demo/StaticCase",
        ]
    )

    assert summarized == [
        "onnxruntime/test/contrib_ops/demo/StaticCase",
        "onnxruntime/test/contrib_ops/demo/TestAlpha_run[0-2]",
        "onnxruntime/test/contrib_ops/demo/TestBeta_run[4-5]",
    ]


def test_render_operator_markdown_renders_both_tables(tmp_path: Path) -> None:
    sample_root = tmp_path / "artifacts"
    case_a_dir = sample_root / "test" / "suite" / "Abs_run0"
    case_a_dir.mkdir(parents=True)
    onnx.save(make_single_operator_model("Abs"), case_a_dir / "model.onnx")
    (case_a_dir / "input_0.pb").write_bytes(b"abc")

    case_b_dir = sample_root / "test" / "suite" / "Abs_run1"
    case_b_dir.mkdir(parents=True)
    onnx.save(make_single_operator_model("Abs"), case_b_dir / "model.onnx")
    (case_b_dir / "output_0.pb").write_bytes(b"defgh")
    (case_b_dir / "validation.json").write_text(
        json.dumps(
            {
                "expects_failure": True,
                "expected_failure_substring": "expected failure",
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )

    case_c_dir = sample_root / "test" / "suite" / "Relu_run0"
    case_c_dir.mkdir(parents=True)
    onnx.save(make_single_operator_model("Relu"), case_c_dir / "model.onnx")
    (case_c_dir / "input_0.pb").write_bytes(b"ijkl")
    (case_c_dir / "validation.json").write_text(
        json.dumps(
            {
                "expects_failure": False,
                "expected_failure_substring": "",
                "excluded_providers": ["CPU"],
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )

    case_d_dir = sample_root / "test" / "suite" / "BiasSoftmax_run0"
    case_d_dir.mkdir(parents=True)
    onnx.save(
        make_single_operator_model("BiasSoftmax", domain="com.microsoft"),
        case_d_dir / "model.onnx",
    )
    (case_d_dir / "input_0.pb").write_bytes(b"mn")
    (case_d_dir / "validation.json").write_text(
        json.dumps(
            {
                "expects_failure": False,
                "expected_failure_substring": "",
                "included_providers": ["CPU"],
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )

    cases = load_single_operator_cases(sample_root)
    abs_onnx_bytes = sum(case.onnx_bytes for case in cases if case.operator == "Abs")
    abs_negative_cases = sum(
        1 for case in cases if case.operator == "Abs" and case.negative_test_case
    )
    bias_softmax_onnx_bytes = sum(
        case.onnx_bytes for case in cases if case.operator == "com.microsoft::BiasSoftmax"
    )
    relu_onnx_bytes = sum(case.onnx_bytes for case in cases if case.operator == "Relu")

    markdown = render_operator_markdown(
        cases,
        repo_root=REPO_ROOT,
        artifacts_root=sample_root,
    )

    assert "## Operator counts" in markdown
    assert "## Operator test cases" in markdown
    assert "## Positive test cases by engine" in markdown
    assert "| Domain | Operator | Test cases | Negative test cases | .onnx bytes | .pb bytes |" in markdown
    assert (
        "| ai.onnx | "
        "[Abs](https://onnx.ai/onnx/operators/onnx__Abs.html) "
        f"| 2 | {abs_negative_cases} | {abs_onnx_bytes} | 8 |"
    ) in markdown
    assert (
        "| com.microsoft | "
        "[BiasSoftmax](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSoftmax) "
        f"| 1 | 0 | {bias_softmax_onnx_bytes} | 2 |"
    ) in markdown
    assert (
        "| ai.onnx | "
        "[Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) "
        f"| 1 | 0 | {relu_onnx_bytes} | 4 |"
    ) in markdown
    assert "test/suite/Abs_run[0-1]" in markdown
    assert (
        "| ai.onnx | "
        "[Abs](https://onnx.ai/onnx/operators/onnx__Abs.html) "
        "| test/suite/Abs_run[0-1] |"
    ) in markdown
    assert (
        "| com.microsoft | "
        "[BiasSoftmax](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSoftmax) "
        "| test/suite/BiasSoftmax_run0 |"
    ) in markdown
    assert (
        "| ai.onnx | "
        "[Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) "
        "| test/suite/Relu_run0 |"
    ) in markdown
    assert "| Domain | Operator | Total positive test cases | CPU |" in markdown
    assert (
        "| ai.onnx | "
        "[Abs](https://onnx.ai/onnx/operators/onnx__Abs.html) "
        "| 1 | 1 |"
    ) in markdown
    assert (
        "| com.microsoft | "
        "[BiasSoftmax](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSoftmax) "
        "| 1 | 1 |"
    ) in markdown
    assert (
        "| ai.onnx | "
        "[Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) "
        "| 1 | 0 :warning: |"
    ) in markdown


def test_render_operator_markdown_includes_multiple_dataset_roots(tmp_path: Path) -> None:
    sample_root = tmp_path / "artifacts"

    positive_case_dir = sample_root / "onnxruntime" / "test" / "suite" / "Abs_run0"
    positive_case_dir.mkdir(parents=True)
    onnx.save(make_single_operator_model("Abs"), positive_case_dir / "model.onnx")

    negative_case_dir = (
        sample_root / "onnxruntime-negative" / "test" / "suite" / "Abs_run1"
    )
    negative_case_dir.mkdir(parents=True)
    onnx.save(make_single_operator_model("Abs"), negative_case_dir / "model.onnx")
    (negative_case_dir / "validation.json").write_text(
        json.dumps(
            {
                "expects_failure": True,
                "expected_failure_substring": "expected failure",
                "outputs": [],
            }
        ),
        encoding="utf-8",
    )

    cases = load_single_operator_cases(sample_root)
    markdown = render_operator_markdown(
        cases,
        repo_root=REPO_ROOT,
        artifacts_root=sample_root,
    )

    assert "Artifact roots:" in markdown
    assert "onnxruntime-negative" in markdown
    assert "onnxruntime/test/suite/Abs_run0" in markdown
    assert "onnxruntime-negative/test/suite/Abs_run1" in markdown


def test_operator_markdown_report() -> None:
    expected_markdown = render_operator_markdown(
        load_single_operator_cases(ARTIFACTS_ROOT),
        repo_root=REPO_ROOT,
        artifacts_root=ARTIFACTS_ROOT,
    )

    if os.environ.get("UPDATE_REFS") == "1":
        REPORT_PATH.write_text(expected_markdown, encoding="utf-8")

    if not REPORT_PATH.exists():
        pytest.fail(
            "artifacts/OPERATORS.md is missing. "
            f"Regenerate with: {DOCS_REGEN_COMMAND}"
        )

    actual_markdown = REPORT_PATH.read_text(encoding="utf-8")
    assert actual_markdown == expected_markdown, (
        "artifacts/OPERATORS.md is stale. "
        f"Regenerate with: {DOCS_REGEN_COMMAND}"
    )
