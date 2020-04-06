from dataclasses import dataclass
from typing import List, Dict

from .PostingListItem import PostingListItem


@dataclass
class TokenIndexItem:
    posting_list: List[PostingListItem]
    term_frequency: Dict[str, int]
    doc_frequency: Dict[str, int]
