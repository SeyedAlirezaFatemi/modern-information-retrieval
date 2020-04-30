from typing import Optional, List

import untangle

from src.enums import Fields
from src.models.Document import Document
from src.models.TextPreparer import TextPreparer
from src.types import DocID
from src.utils.create_doc import create_doc_from_page


def read_document(
    docs_path: str, doc_id: DocID, text_preparer: TextPreparer
) -> Optional[Document]:
    tree = untangle.parse(docs_path)
    document = None
    for page in tree.mediawiki.page:
        if DocID(page.id.cdata) == doc_id:
            document = create_doc_from_page(page, text_preparer)
            break
    return document


def read_documents_json(
    docs_path: str, text_preparer: TextPreparer, fields: Optional[List[Fields]] = None
) -> List[Document]:
    if fields is None:
        fields = list(Fields)
    import json

    with open(docs_path) as f:
        data = json.load(f)

    return data
