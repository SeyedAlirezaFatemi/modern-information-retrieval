import sys

from src.enums import Fields
from src.models.Manager import Manager
from src.models.TextPreprocessor import PersianTextPreprocessor
from .create_documents import create_documents

sys.setrecursionlimit(10 ** 6)


def construct_positional_indexes(
    docs_path: str = "./data/Persian.xml", fields=None, multiprocess: bool = False
) -> Manager:
    if fields is None:
        fields = [Fields.TEXT, Fields.TITLE]
    text_preprocessor = PersianTextPreprocessor()
    documents = create_documents(docs_path, text_preprocessor, multiprocess)
    return Manager(documents, fields, text_preprocessor)
