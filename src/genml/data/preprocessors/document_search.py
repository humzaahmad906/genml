from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.utils import convert_files_to_docs, fetch_archive_from_http


def create_subdocs(doc_dir, additional_meta):
    """
    Create Subdocs from multiple documents stored in doc_dir directory and add additional meta
    :param doc_dir: directory where documents are present
    :param additional_meta: additional meta that needs to be added
    :return: subdocs
    """

    all_docs = convert_files_to_docs(dir_path=doc_dir)
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
    changed_docs = []
    for doc in docs:
        doc[["meta"]] = {**doc["meta"], **additional_meta}
        changed_docs.append(doc)
    del docs
    return changed_docs
