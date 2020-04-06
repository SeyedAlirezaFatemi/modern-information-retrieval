from enum import Enum


class Fields(Enum):
    TITLE = "title"
    TEXT = "text"


FIELDS = [field.value for field in list(Fields)]
