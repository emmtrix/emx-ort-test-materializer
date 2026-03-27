<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_artifact_validation_docs.py::test_artifact_validation_error_doc -->

# ORT artifact validation errors

Aggregates non-OK artifact validation outcomes.

Expectation source: `tests/artifact_validation_expected.json`

Validated cases: 4182 / 4184 OK, 2 non-OK.

Ignored artifact generation cases: 5.

| Error message | Count | Sources |
| --- | --- | --- |
| ONNX Runtime error | 2 | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test |

## Error frequency by source

| Error message | Source | Count |
| --- | --- | --- |
| ONNX Runtime error | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | 2 |

## Failing artifact cases

Lists every artifact case with a non-OK expected validation result.

| Case | Source | Error |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DBlkZp_run0 | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | [ONNXRuntimeError] : 1 : FAIL : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DBlkZp_run0/model.onnx failed:Node (node1) Op (MatMulFpQ4) [ShapeInferenceError] 4b quantization not yet supported on this hardware platform! |
| artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DSym_run0 | artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test | [ONNXRuntimeError] : 1 : FAIL : Load model from D:/emmtrix/git/emx-ort-test-materializer/artifacts/onnxruntime/test/contrib_ops/matmul_fpq4_test/MatMul2DSym_run0/model.onnx failed:Node (node1) Op (MatMulFpQ4) [ShapeInferenceError] 4b quantization not yet supported on this hardware platform! |

## Ignored Artifact Generation Cases

Lists configured artifact cases that generation skips, together with the tracked reason.

| Case | Source | Reason |
| --- | --- | --- |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_BFloat16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline can replay this legacy LayerNormalization bfloat16 case without surfacing a known CPU environment limitation as an artifact failure. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16ScaleBiasOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16Input_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
| artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16ScaleOutput_run0 | artifacts/onnxruntime/test/contrib_ops/layer_norm_op_test | Ignored until the runtime artifact pipeline preserves a compatible ONNX opset import for legacy mixed-precision LayerNormalization exports. |
