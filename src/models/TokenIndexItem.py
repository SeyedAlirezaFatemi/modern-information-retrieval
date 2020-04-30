from dataclasses import dataclass
from typing import List, Dict

from .PostingListItem import PostingListItem
from ..enums import Fields


@dataclass
class TokenIndexItem:
    posting_list: List[PostingListItem]
    term_frequency: Dict[Fields, int]
    doc_frequency: Dict[Fields, int]
