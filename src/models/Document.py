from typing import List, Optional, Dict

from src.enums import Fields
from src.types import DocID, Token
from .TextPreprocessor import TextPreprocessor


class Document:
    def __init__(
        self,
        text_preprocessor: TextPreprocessor,
        doc_id: DocID,
        data: Dict[Fields, str],
        category: Optional[int] = None,
    ):
        self.doc_id = doc_id
        self.data = data
        self.data_tokens = dict()
        for filed, value in data.items():
            self.data_tokens[filed] = text_preprocessor.preprocess_text(value)
        self.category = category

    def get_tokens(self, field: Fields) -> List[Token]:
        return self.data_tokens[field]

    def get_field(self, field: Fields) -> str:
        return self.data[field]

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.doc_id == self.doc_id
        else:
            return False
