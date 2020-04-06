from dataclasses import dataclass
from typing import List, Dict

from src.models import PostingListItem


@dataclass
class TokenIndexItem:
    posting_list: List[PostingListItem]
    term_frequency: Dict[str, int]
    doc_frequency: Dict[str, int]
