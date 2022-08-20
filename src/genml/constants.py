import yaml


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


QA_CONFIG_PATH = "model_configs/nlp/question_answering.yaml"
QA_CONFIG = read_config(QA_CONFIG_PATH)
DEEPSET_CONFIG = {**QA_CONFIG, "model_id": QA_CONFIG["model_id"][0]}


FE_CONFIG_PATH = "model_configs/nlp/feature_extraction.yaml"
FE_CONFIG = read_config(FE_CONFIG_PATH)
CODEBERT_CONFIG = {**FE_CONFIG, "model_id": FE_CONFIG["model_id"][0]}


SUBDOCS_DIRECTORY = "data/documents"
TEMP_FILE_UPLOADED_NAME = "data/file_upload_directory/temp.pdf"

FAISS_INDEX_FILENAME = "data/faiss/faiss.index"
FAISS_TRAINING_DATA = "data/faiss/training_data.json"
