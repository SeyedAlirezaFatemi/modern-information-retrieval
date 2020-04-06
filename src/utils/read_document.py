from typing import Optional

import untangle

from src.models import Document, TextPreparer
from src.types import DocID


def read_document(
    docs_path: str, doc_id: DocID, text_preparer: TextPreparer
) -> Optional[Document]:
    tree = untangle.parse(docs_path)
    document = None
    for page in tree.mediawiki.page:
        if DocID(page.id.cdata) == doc_id:
            document = Document(
                text_preparer,
                DocID(page.id.cdata),
                page.title.cdata,
                page.revision.text.cdata,
            )
            break
    return document
