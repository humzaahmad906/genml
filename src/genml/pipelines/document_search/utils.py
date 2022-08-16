import os
from pathlib import Path
import shutil

from haystack.nodes import PreProcessor
from haystack.utils import convert_files_to_docs

from src.genml.constants import SUBDOCS_DIRECTORY, TEMP_FILE_UPLOADED_NAME


class SubDocsPreprocessor:
    def __init__(self, metadata):
        self.docs = None
        self.doc_dir = SUBDOCS_DIRECTORY
        self.extra_meta = metadata

    def create_sub_docs(self):
        all_docs = convert_files_to_docs(dir_path=self.doc_dir)
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=100,
            split_respect_sentence_boundary=True
        )
        docs = preprocessor.process(all_docs)
        docs = [doc.__dict__ for doc in docs]
        extra_meta_docs = []
        for doc in docs:
            doc[["meta"]] = {**doc["meta"], **self.extra_meta}
            extra_meta_docs.append(doc)
        del docs
        self.docs = extra_meta_docs

    def remove_all_docs(self):
        self.docs = None
        for f in os.listdir(self.doc_dir):
            os.remove(os.path.join(self.doc_dir, f))

    def get_present_docs(self):
        return self.docs

    def __call__(self, document_path):
        self.remove_all_docs()
        document_path = Path(document_path)
        save_path = os.path.join(self.doc_dir, document_path.name)
        os.rename(document_path, save_path)
        create_sub_docs()
        return self.docs


def write_uploaded_file(upload_file):
    try:
        if os.path.exits(TEMP_FILE_UPLOADED_NAME):
            os.remove(TEMP_FILE_UPLOADED_NAME)
        with open(TEMP_FILE_UPLOADED_NAME, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


