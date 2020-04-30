import multiprocessing
from typing import List

import untangle
from tqdm import tqdm

from src.models.Document import Document
from src.models.TextPreparer import TextPreparer
from src.types import DocID
from .create_doc import create_doc_from_page


def create_documents(
    docs_path: str, text_preparer: TextPreparer, multiprocess: bool = False
) -> List[Document]:
    tree = untangle.parse(docs_path)
    documents = []

    if multiprocess:
        pool = multiprocessing.Pool()
        for page in tree.mediawiki.page:
            documents.append(
                pool.apply_async(create_doc_from_page, args=(page, text_preparer))
            )
        pool.close()
        pool.join()
        documents = [res.get() for res in documents]
    else:
        for page in tqdm(tree.mediawiki.page):
            documents.append(create_doc_from_page(page, text_preparer))
    documents = sorted(documents, key=lambda document: document.doc_id)
    return documents
