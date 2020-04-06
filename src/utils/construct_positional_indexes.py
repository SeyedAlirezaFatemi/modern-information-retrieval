import sys

from src.models import TextPreparer, Manager
from src.utils import create_documents

sys.setrecursionlimit(10 ** 6)


def construct_positional_indexes(
    docs_path: str = "./data/Persian.xml", multiprocess: bool = False
) -> Manager:
    text_preparer = TextPreparer()
    documents = create_documents(docs_path, text_preparer, multiprocess)
    return Manager(documents, text_preparer)
