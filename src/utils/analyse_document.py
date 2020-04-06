from typing import Dict, Tuple

from src.enums import FIELDS
from src.models import Document, PostingListItem
from src.types import Token


def analyse_document(
    document: Document
) -> Tuple[Dict[Token, PostingListItem], Dict[str, Dict[Token, int]]]:
    token_positional_list_item_dict = dict()
    token_frequency_dict = dict()
    for field in FIELDS:
        token_frequency_dict[field] = dict()
        for idx, token in enumerate(document[f"{field}_tokens"]):
            if token not in token_positional_list_item_dict:
                token_positional_list_item_dict[token] = PostingListItem(
                    document.doc_id
                )
            if token not in token_frequency_dict[field]:
                token_frequency_dict[field][token] = 0
            token_positional_list_item_dict[token].add_to_positions(field, idx)
            token_frequency_dict[field][token] += 1
    return token_positional_list_item_dict, token_frequency_dict
