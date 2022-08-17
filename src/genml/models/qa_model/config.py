from pathlib import Path

qa_onnx_path = Path("onnx_qa")
task = "question-answering"

DEEPSET_CONFIG = {
    "onnx_path": qa_onnx_path,
    "model_id": "deepset/roberta-base-squad2",
    "task": task,
    "model_filename": "model.onnx",
    "optimized_filename": "model-optimized.onnx",
    "quantized_filename": "model-quantized.onnx",
}
