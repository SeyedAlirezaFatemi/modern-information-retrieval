from src.enums import FIELDS
from src.models.TokenIndexItem import TokenIndexItem


def create_new_token_index_item() -> TokenIndexItem:
    term_frequency = dict()
    doc_frequency = dict()
    for field in FIELDS:
        term_frequency[field] = 0
        doc_frequency[field] = 0
    return TokenIndexItem([], term_frequency, doc_frequency)
