import faiss
import json

from src.genml.constants import FAISS_INDEX_FILENAME, FAISS_TRAINING_DATA


class Faiss:
    def __init__(self, from_file=False):
        if from_file:
            self.index = faiss.read_index(FAISS_INDEX_FILENAME)
        else:
            m = 8  # number of centroid IDs in final compressed vectors
            bits = 8  # number of bits in each centroid
            # quantizer = faiss.IndexFlatIP(d) for inner product as similarity
            quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
            self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
            self.train()
        self.index.nprobe = 10
        self.k = 5

    def train(self):
        embeddings_data = json.load(open(FAISS_TRAINING_DATA, 'r'))
        self.index.train(embeddings_data)

    def write_index(self):
        faiss.write_index(self.index, FAISS_INDEX_FILENAME)

    def search(self, xq):
        _, indices = self.index.search(xq, self.k)
        print(indices)

