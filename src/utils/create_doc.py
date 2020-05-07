from typing import Dict

from src.enums import Fields
from src.models.Document import Document
from src.models.TextPreprocessor import TextPreprocessor
from src.types import DocID


def create_doc_from_page(
    page, text_preprocessor: TextPreprocessor, debug: bool = False
) -> Document:
    if debug:
        print(f"Creation of document {page.id.cdata} started!")
    doc_data = {Fields.TITLE: page.title.cdata, Fields.TEXT: page.revision.text.cdata}
    doc = Document(text_preprocessor, DocID(page.id.cdata), doc_data)
    if debug:
        print(f"Document {page.id.cdata} created!")
    return doc


def create_doc_from_json(
    raw_document: Dict, doc_id: DocID, text_preprocessor: TextPreprocessor
) -> Document:
    category = raw_document["category"]
    doc_data = {Fields.BODY: raw_document["body"], Fields.TITLE: raw_document["title"]}
    return Document(text_preprocessor, doc_id, doc_data, category)
