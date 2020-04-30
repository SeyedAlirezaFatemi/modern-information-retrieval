from typing import List

from src.enums import Fields
from src.types import DocID


class PostingListItem:
    def __init__(self, doc_id: DocID, fields: List[Fields]):
        self.doc_id = doc_id
        self.fields = fields
        for field in fields:
            self[f"{field.value}_positions"] = []
            self[f"{field.value}_tf"] = 0

    def add_to_positions(self, field: Fields, position: int):
        self[f"{field.value}_positions"].append(position)
        self[f"{field.value}_tf"] += 1

    def get_positions(self, field: Fields) -> List[int]:
        return self[f"{field.value}_positions"]

    def get_tf(self, field: Fields) -> int:
        return self[f"{field.value}_tf"]

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.doc_id == self.doc_id
        else:
            return False

    def __str__(self):
        return f"""
        Document ID: {self.doc_id}
        Title Positions: {self.get_positions(Fields.TITLE)[:10]}
        Text Positions: {self.get_positions(Fields.TEXT)[:10]}
        """
