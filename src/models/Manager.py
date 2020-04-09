import pickle
import re
from typing import List, Dict, Union, Set

import numpy as np

from src.enums import Methods, Fields
from src.types import DocID, Token
from src.utils.read_document import read_document
from .BigramIndex import BigramIndex
from .CorpusIndex import CorpusIndex
from .Document import Document
from .TextPreparer import TextPreparer

text_preparer = TextPreparer()


class Manager:
    documents: Dict[DocID, Document]

    def __init__(self, documents: List[Document]):
        self.documents = dict()
        for document in documents:
            self.documents[document.doc_id] = document
        self.corpus_index = CorpusIndex(documents)
        self.bigram_index = BigramIndex(self.corpus_index)

    def add_document_to_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document already exists
        if doc_id in self.documents:
            print("Document already exists!")
            return
        # Read document
        document = read_document(docs_path, doc_id, text_preparer)
        if document is None:
            print("Document not found!")
            return
        # Find where we must put the documents in self.documents list
        self.documents[doc_id] = document
        self.corpus_index.add_document_to_indexes(document)

    def delete_document_from_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document exists
        if doc_id not in self.documents:
            print("Document does not exists!")
            return
        document = read_document(docs_path, doc_id, text_preparer)
        if document is None:
            print("Document not found!")
            return
        del self.documents[doc_id]
        self.corpus_index.delete_document_from_indexes(document)

    def search_phrasal(
        self,
        tokenized_phrase: List[Token],
        field: Fields,
        prev_candidates: Set[DocID],
        strict: bool = False,
        first_phrasal: bool = False,
    ) -> Set[DocID]:
        postings = [
            self.corpus_index.get_posting_list(token) for token in tokenized_phrase
        ]
        pointers = [0] * len(postings)
        current_candidates = set()
        while True:
            try:
                pointed_doc_ids = [
                    postings[idx][pointer].doc_id
                    for idx, pointer in enumerate(pointers)
                ]
            except IndexError:
                break
            min_idx = np.argmin(pointed_doc_ids)
            # Check if all pointers are pointing to the same doc
            unique_pointed_docs = set(pointed_doc_ids)
            if len(unique_pointed_docs) != 1:
                # Move the pointer pointing to the minimum doc to the next doc
                pointers[min_idx] += 1
                continue
            else:
                doc_id = unique_pointed_docs.pop()
                # Check correct positions
                posting_items = [
                    postings[idx][pointer] for idx, pointer in enumerate(pointers)
                ]
                # Compare other positions to ref
                ref = posting_items[0]
                # Check the exact phrase by checking the positions
                for position in ref.get_positions(field):
                    count = 0
                    for posting_idx, other_posting_items in enumerate(
                        posting_items[1:]
                    ):
                        if (
                            position + posting_idx + 1
                            in other_posting_items.get_positions(field)
                        ):
                            count += 1
                    if count == len(posting_items) - 1:
                        if strict and not first_phrasal:
                            if doc_id in prev_candidates:
                                current_candidates.add(doc_id)
                        else:
                            current_candidates.add(doc_id)
                        break
                # Move all pointers to the next doc
                pointers = [point + 1 for point in pointers]
        if strict:
            return current_candidates
        return prev_candidates.union(current_candidates)

    def search(
        self,
        field_queries: Union[Dict[Fields, str], str],
        field_weights: Dict[Fields, float] = None,
        method: Methods = Methods.LTN_LNN,
        max_retrieved: int = 15,
        strict: bool = False,
    ) -> List[DocID]:
        if method != Methods.LTC_LNC and method != Methods.LTN_LNN:
            print(f"Method {method} is not supported!")
            return []
        if isinstance(field_queries, str):
            query = field_queries
            field_queries = dict()
            for field in Fields:
                field_queries[field] = query
        if field_weights is None:
            field_weights = dict()
            for field in field_queries:
                field_weights[field] = 1.0
        # Docs which satisfy phrasal search if there is any
        candidate_docs = set()
        # Initiate scores
        scores = dict()
        for field in field_queries:
            scores[field] = dict()
        normalized_scores = dict()
        # Docs with positive scores
        all_positive_docs = set()
        phrasal = False
        for field in field_queries:
            query = field_queries[field]
            # Extract phrases if there are any
            phrases = re.findall(r'"([^"]*)"', query)
            tokenized_phrases = [
                text_preparer.prepare_text(phrase) for phrase in phrases
            ]

            if len(tokenized_phrases) != 0:
                phrasal = True
            candidate_docs = set()
            first_phrasal = True
            for tokenized_phrase in tokenized_phrases:
                candidate_docs = self.search_phrasal(
                    tokenized_phrase, field, candidate_docs, True, first_phrasal
                )
                first_phrasal = False
            # Remove "
            query.replace('"', " ")
            query_tokens = text_preparer.prepare_text(query)

            token_and_weights = list(
                (
                    (token, 1 + np.log10(tf))
                    for token, tf in zip(
                        *list(
                            list(x) for x in np.unique(query_tokens, return_counts=True)
                        )
                    )
                )
            )

            num_docs = len(self.documents)

            for query_token, token_weight in token_and_weights:
                posting_list = self.corpus_index.get_posting_list(query_token)
                if len(posting_list) == 0:
                    continue
                df = self.corpus_index.index[query_token].doc_frequency[field]
                if df == 0:
                    continue
                idf = np.log10(num_docs / df)
                for posting_list_item in posting_list:
                    doc_id = posting_list_item.doc_id
                    # If doing phrasal search, check for candidate docs
                    if phrasal and doc_id not in candidate_docs:
                        continue
                    tf = posting_list_item.get_tf(field)
                    if tf == 0:
                        continue
                    w = 1 + np.log10(tf)
                    if doc_id not in scores[field]:
                        scores[field][doc_id] = [0.0, 0.0]
                        # Calculate weights for tokens not in query for normalization
                        if method == Methods.LTC_LNC:
                            for non_query_token, non_query_token_w in list(
                                (
                                    (token, 1 + np.log10(tf))
                                    for token, tf in zip(
                                        *list(
                                            np.unique(
                                                self.documents[doc_id].get_tokens(
                                                    field
                                                ),
                                                return_counts=True,
                                            )
                                        )
                                    )
                                )
                            ):
                                if non_query_token not in query_tokens:
                                    non_query_token_df = self.corpus_index.index[
                                        non_query_token
                                    ].doc_frequency[field]
                                    if non_query_token_df == 0:
                                        continue
                                    non_query_token_idf = np.log10(
                                        num_docs / non_query_token_df
                                    )
                                    scores[field][doc_id][1] += (
                                        non_query_token_idf * non_query_token_w
                                    ) ** 2

                    all_positive_docs.add(doc_id)
                    scores[field][doc_id][0] += token_weight * idf * w
                    scores[field][doc_id][1] += (idf * w) ** 2

            # Normalize scores
            normalized_scores[field] = dict()
            for doc_id in scores[field]:
                normalized_scores[field][doc_id] = (
                    scores[field][doc_id][0] / np.sqrt(scores[field][doc_id][1])
                    if method == Methods.LTC_LNC
                    else scores[field][doc_id][0]
                )

        # Apply field weights
        final_scores = []
        for doc_id in all_positive_docs:
            score = 0
            for field in field_queries:
                if doc_id in normalized_scores[field]:
                    score += normalized_scores[field][doc_id] * field_weights[field]
            final_scores.append((doc_id, score))
        # Sort final scores
        final_scores.sort(key=lambda item: -item[1])

        relevant_docs = list(doc_id for doc_id, score in final_scores)
        return relevant_docs[:max_retrieved]

    def save_index(self, destination: str) -> None:
        with open(destination, "wb") as f:
            pickle.dump(self, f)
