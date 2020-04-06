from src.enums import FIELDS
from src.types import DocID


class PostingListItem:
    def __init__(self, doc_id: DocID):
        self.doc_id = doc_id
        for field in FIELDS:
            self[f"{field}_positions"] = []
            self[f"{field}_tf"] = 0

    def add_to_positions(self, field: str, position: int):
        self[f"{field}_positions"].append(position)
        self[f"{field}_tf"] += 1

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __str__(self):
        return f"""
        Document ID: {self.doc_id}
        Title Positions: {self.title_positions[:10]}
        Text Positions: {self.text_positions[:10]}
        """
