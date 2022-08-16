from fastapi import APIRouter, UploadFile
import logging

from src.genml.pipelines.document_search.utils import write_uploaded_file, SubDocsPreprocessor


logging = logging.getLogger(__name__)


subdocs_preprocessor = SubDocsPreprocessor()

document_search = APIRouter()


@document_search.post("/index_document", response_model=RelevancyUrlResponse)
def index_document(upload_file: UploadFile):
    write_uploaded_file(upload_file)
    subdocs = subdocs_preprocessor()

    return RelevancyUrlResponse(**response_json)


@document_search.post("/search_relevant_document", response_model=DeleteESResponse)
def search_relevant_document(request: DeleteESRequest):
    """
    Delete file from Elastic, Mongo DB and Bucket
    Args:
        request (request):  Request object
    Return:
        response (Dict): response after deletion pipeline run
    """

    data_list = request.data
    google_storage = GoogleCloudStorage()
    if len(data_list) != 0:
        response = None
        connection = connect_elastic()
        for data in data_list:
            try:
                data_tuple = extract_relevant_info(data)
                state_name = data_tuple[1]
                document_id = data_tuple[2]
                downloaded_doc_link = data_tuple[10]
                raw_text_link = data_tuple[11]
                style_text_link = data_tuple[12]

                if downloaded_doc_link is not None:
                    downloaded_filename = downloaded_doc_link.split(BUCKET_NAME)[1][1:]
                    google_storage.delete_file_from_bucket(downloaded_filename)
                if raw_text_link is not None:
                    raw_text_filename = raw_text_link.split(BUCKET_NAME)[1][1:]
                    google_storage.delete_file_from_bucket(raw_text_filename)
                if style_text_link is not None:
                    style_text_filename = style_text_link.split(BUCKET_NAME)[1][1:]
                    google_storage.delete_file_from_bucket(style_text_filename)

                is_delete, acknowledged, deleted_count = delete_doc(
                    state_name, document_id
                )
                logging.info(
                    f"is_delete={is_delete}, acknowledged={acknowledged}, deleted_count={deleted_count}"  # noqa
                )

                search_body = {"query": {"match": {"doc_id": document_id}}}
                query_response3 = connection.search(index=state_name, body=search_body)
                ids_to_delete = []
                for hit in query_response3["hits"]["hits"]:
                    ids_to_delete.append(hit["_id"])
                for id_to_delete in ids_to_delete:
                    connection.delete(index=state_name, id=id_to_delete)
                logging.info(f"Id: {document_id} deleted from Elastic Search Database")
            except Exception as e:
                logging.error(e)
                response = {
                    "error": True,
                    "message": f"Could not delete entries because {e}",
                }
        if response is None:
            response = {
                "error": False,
                "message": "successfully deleted entries from elastic search",
            }
    else:
        response = {"error": True, "message": "did not receive data"}
    return RelevancyUrlResponse(**response)
