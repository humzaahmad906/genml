from pathlib import Path

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig

from src.genml.models.qa_model.config import DEEPSET_CONFIG


class QADeepset:
    def __init__(self):
        self.config = DEEPSET_CONFIG
        self.onnx_path = self.config["onnx_path"]
        self.model_id = self.config["onnx_path"]
        self.task = self.config["task"]
        self.model_filename = self.config["model_filename"]
        self.optimized_filename = self.config["optimized_filename"]
        self.quantized_filename = self.config["quantized_filename"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def load_quantized_model(self):
        qt_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_path, file_name=self.quantized_filename)
        # test the quantized model with using transformers pipeline
        quantized_qa_model = pipeline(self.task, model=qt_model, tokenizer=self.tokenizer, handle_impossible_answer=True)
        return quantized_qa_model

    def load_optimized_model(self):
        # load quantized model
        opt_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_path, file_name=self.optimized_filename)

        # test the quantized model with using transformers pipeline
        optimum_qa_model = pipeline(self.task, model=opt_model, tokenizer=self.tokenizer, handle_impossible_answer=True)
        return optimum_qa_model

    def create_optimized_model(self):
        # create ORTOptimizer and define optimization configuration
        optimizer = ORTOptimizer.from_pretrained(self.model_id, feature=self.task)
        optimization_config = OptimizationConfig(optimization_level=99) # enable all optimizations

        # apply the optimization configuration to the model
        optimizer.export(
            onnx_model_path=self.onnx_path / self.model_filename,
            onnx_optimized_model_output_path=self.onnx_path / self.optimized_filename,
            optimization_config=optimization_config,
        )

    def create_quantized_model(self):
        self.create_optimized_model()
        # create ORTQuantizer and define quantization configuration
        quantizer = ORTQuantizer.from_pretrained(self.model_id, feature=self.task)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

        # apply the quantization configuration to the model
        quantizer.export(
            onnx_model_path=self.onnx_path / self.optimized_filename,
            onnx_quantized_model_output_path=self.onnx_path / self.quantized_filename,
            quantization_config=qconfig,
        )



