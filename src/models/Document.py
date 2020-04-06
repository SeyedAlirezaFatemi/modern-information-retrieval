from typing import List

from src.prepare_text import TextPreparer
from src.types import Token, DocID


class Document:
    title_tokens: List[Token]
    text_tokens: List[Token]

    def __init__(
        self, text_preparer: TextPreparer, doc_id: DocID, title: str, text: str
    ):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.title_tokens = text_preparer.prepare_text(title)
        self.text_tokens = text_preparer.prepare_text(text)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)
