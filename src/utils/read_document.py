from typing import Optional, List

import untangle

from src.enums import Fields
from src.models.Document import Document
from src.models.TextPreprocessor import TextPreprocessor
from src.types import DocID
from src.utils.create_doc import create_doc_from_page, create_doc_from_json


def read_document(
    docs_path: str, doc_id: DocID, text_preprocessor: TextPreprocessor
) -> Optional[Document]:
    tree = untangle.parse(docs_path)
    document = None
    for page in tree.mediawiki.page:
        if DocID(page.id.cdata) == doc_id:
            document = create_doc_from_page(page, text_preprocessor)
            break
    return document


def read_documents_json(
    docs_path: str,
    text_preprocessor: TextPreprocessor,
    fields: Optional[List[Fields]] = None,
) -> List[Document]:
    if fields is None:
        fields = list(Fields)
    import json

    with open(docs_path) as f:
        raw_documents = json.load(f)
    documents = []
    for index, raw_document in enumerate(raw_documents):
        documents.append(create_doc_from_json(raw_document, index, text_preprocessor))

    return documents
