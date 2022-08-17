from pathlib import Path

fe_onnx_path = Path("onnx_fe")
task = "feature-extraction"

CODEBERT_CONFIG = {
    "onnx_path": fe_onnx_path,
    "model_id": "microsoft/codebert-base",
    "task": task,
    "model_filename": "model.onnx",
    "optimized_filename": "model-optimized.onnx",
    "quantized_filename": "model-quantized.onnx",
}