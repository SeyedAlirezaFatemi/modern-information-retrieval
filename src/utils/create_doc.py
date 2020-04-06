from src.models.Document import Document
from src.models.TextPreparer import TextPreparer
from src.types import DocID


def create_doc(page, text_preparer: TextPreparer, debug: bool = False) -> Document:
    if debug:
        print(f"Creation of document {page.id.cdata} started!")
    doc = Document(
        text_preparer, DocID(page.id.cdata), page.title.cdata, page.revision.text.cdata
    )
    if debug:
        print(f"Document {page.id.cdata} created!")
    return doc
