from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

from src.genml.utils.faiss_indexing import Faiss

model_id = "microsoft/codebert-base"
onnx_path = Path("onnx")
task = "feature-extraction"

# load vanilla transformers and convert to onnx
model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# # save onnx checkpoint and tokenizer
# model.save_pretrained(onnx_path)
# tokenizer.save_pretrained(onnx_path)

# test the model with using transformers pipeline, with handle_impossible_answer for squad_v2
Faiss.create_training_data(model, tokenizer)


faiss = Faiss(768)
