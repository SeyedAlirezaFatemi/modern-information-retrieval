import sys

from src.models.Manager import Manager
from src.models.TextPreparer import TextPreparer
from .create_documents import create_documents

sys.setrecursionlimit(10 ** 6)


def construct_positional_indexes(
    docs_path: str = "./data/Persian.xml", multiprocess: bool = False
) -> Manager:
    text_preparer = TextPreparer()
    documents = create_documents(docs_path, text_preparer, multiprocess)
    return Manager(documents)
