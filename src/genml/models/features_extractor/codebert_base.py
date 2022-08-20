from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForFeatureExtraction

from src.genml.constants import CODEBERT_CONFIG


class CodeBert:
    def __init__(self):
        self.config = CODEBERT_CONFIG
        self.model_id = self.config["model_id"]
        self.task = self.config["task"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def load_optimized_model(self):
        # load vanilla transformers and convert to onnx
        model = ORTModelForFeatureExtraction.from_pretrained(self.model_id, from_transformers=True)

        # test the model with using transformers pipeline, with handle_impossible_answer for squad_v2
        optimum_feature_extractor = pipeline(self.task, model=model, tokenizer=self.tokenizer, handle_impossible_answer=True)
        return optimum_feature_extractor
