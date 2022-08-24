import numpy as np
import torch.nn.functional as F
import json
from tqdm import tqdm

import faiss

from src.genml.constants import (
    FAISS_INDEX_FILENAME,
    FAISS_TRAINING_DATA,
    INDEX_TRAINING_DOCS_DIRECTORY,
)
from src.genml.data.preprocessors.document_search import create_subdocs
from src.genml.utils.feature_extraction import mean_pooling


class Faiss:
    def __init__(self, d=128, nlist=316, from_file=False):
        if from_file:
            self.index = faiss.read_index(FAISS_INDEX_FILENAME)
        else:
            m = 8  # number of centroid IDs in final compressed vectors
            bits = 8  # number of bits in each centroid
            # quantizer = faiss.IndexFlatIP(d) # for inner product as similarity
            quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
            self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
            self.train()
        self.index.nprobe = 10
        self.k = 5

    def train(self):
        embeddings_data = json.load(open(FAISS_TRAINING_DATA, "r"))
        self.index.train(np.array(embeddings_data, np.float32))
        self.add_index(np.array(embeddings_data, np.float32))

    def add_index(self, sentence_embeddings):
        self.index.add(sentence_embeddings)

    def write_index(self):
        faiss.write_index(self.index, FAISS_INDEX_FILENAME)

    def search(self, xq):
        _, indices = self.index.search(xq, self.k)
        return indices

    @staticmethod
    def create_training_data(model, tokenizer):
        subdocs = create_subdocs(INDEX_TRAINING_DOCS_DIRECTORY)
        training_data = []
        for subdoc in tqdm(subdocs):
            try:
                text = subdoc["content"]
                encoded_input = tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                )
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                prediction = F.normalize(sentence_embeddings, p=2, dim=1).tolist()
                training_data += prediction
            except Exception as e:
                print(e)
                pass
        json.dump(training_data, open(FAISS_TRAINING_DATA, "w"))
