from typing import List, Dict

from tqdm import tqdm

from src.enums import FIELDS
from src.models import Document, PostingListItem, TokenIndexItem
from src.types import Token
from src.utils import (
    next_greater,
    binary_search,
    create_new_token_index_item,
    analyse_document,
)


class CorpusIndex:
    def __init__(self, documents: List[Document]):
        self.index = self.construct_index(documents)

    def construct_index(self, documents: List[Document]) -> Dict[Token, TokenIndexItem]:
        corpus_index = dict()
        for document in tqdm(documents):
            token_positional_list_item_dict, token_frequency_dict = analyse_document(
                document
            )
            for token in token_positional_list_item_dict:
                if token not in corpus_index:
                    corpus_index[token] = create_new_token_index_item()
                corpus_index[token].posting_list.append(
                    token_positional_list_item_dict[token]
                )
            for field in FIELDS:
                for token in token_frequency_dict[field]:
                    corpus_index[token].term_frequency[field] += token_frequency_dict[
                        field
                    ][token]
                    corpus_index[token].doc_frequency[field] += 1

        return corpus_index

    def add_document_to_indexes(self, document: Document) -> None:
        # Analyse document
        token_positional_list_item_dict, token_frequency_dict = analyse_document(
            document
        )
        for token in token_positional_list_item_dict:
            if token not in self.index:
                self.add_token_to_index(token)
                self.index[token].posting_list.append(
                    token_positional_list_item_dict[token]
                )
            else:
                insertion_idx = next_greater(
                    self.index[token].posting_list, document, key=lambda x: x.doc_id
                )
                insertion_idx = (
                    insertion_idx
                    if insertion_idx != -1
                    else len(self.index[token].posting_list)
                )
                self.index[token].posting_list.insert(
                    insertion_idx, token_positional_list_item_dict[token]
                )
        # Update doc and term frequencies
        for field in FIELDS:
            for token in token_frequency_dict[field]:
                self.index[token].term_frequency[field] += token_frequency_dict[field][
                    token
                ]
                self.index[token].doc_frequency[field] += 1
        return

    def delete_document_from_indexes(self, document: Document) -> None:
        token_positional_list_item_dict, token_frequency_dict = analyse_document(
            document
        )
        for token in token_positional_list_item_dict:
            if token not in self.index:
                continue
            else:
                deletion_idx = binary_search(
                    self.index[token].posting_list, document, key=lambda x: x.doc_id
                )
                if deletion_idx == -1:
                    continue
                del self.index[token].posting_list[deletion_idx]
        # Update doc and term frequencies
        for field in FIELDS:
            for token in token_frequency_dict[field]:
                self.index[token].doc_frequency[field] -= 1
                self.index[token].term_frequency[field] -= token_frequency_dict[field][
                    token
                ]
        # Delete tokens with empty posting list
        for token in token_positional_list_item_dict:
            if len(self.index[token].posting_list) == 0:
                del self.index[token]

    def get_posting_list(self, token: Token) -> List[PostingListItem]:
        return self.index[token].posting_list

    def add_token_to_index(self, token: str) -> None:
        self.index[token] = create_new_token_index_item()
