from typing import Dict, Tuple, List

from src.enums import Fields
from src.models.Document import Document
from src.models.PostingListItem import PostingListItem
from src.types import Token


def analyse_document(
    document: Document, fields: List[Fields]
) -> Tuple[Dict[Token, PostingListItem], Dict[Fields, Dict[Token, int]]]:
    token_positional_list_item_dict = dict()
    token_frequency_dict = dict()
    for field in fields:
        token_frequency_dict[field] = dict()
        for idx, token in enumerate(document.get_tokens(field)):
            if token not in token_positional_list_item_dict:
                token_positional_list_item_dict[token] = PostingListItem(
                    document.doc_id, fields
                )
            if token not in token_frequency_dict[field]:
                token_frequency_dict[field][token] = 0
            token_positional_list_item_dict[token].add_to_positions(field, idx)
            token_frequency_dict[field][token] += 1
    return token_positional_list_item_dict, token_frequency_dict
