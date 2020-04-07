import pickle
import re
from typing import List, Dict

import numpy as np

from src.enums import FIELDS, Methods
from src.types import DocID
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

    def search(
        self, query: str, method: str = "ltn-lnn", max_retrieved: int = 15, weights=None
    ) -> List[DocID]:
        if weights is None:
            weights = [1.0, 2.0]
        if method != Methods.LTC_LNC.value and method != Methods.LTN_LNN.value:
            print(f"Method {method} is not supported!")
            return []

        # Extract phrases if there are any
        phrases = re.findall(r'"([^"]*)"', query)
        tokenized_phrases = [text_preparer.prepare_text(phrase) for phrase in phrases]
        phrasal = False
        if len(tokenized_phrases) != 0:
            phrasal = True
        candidate_docs = set()

        for tokenized_phrase in tokenized_phrases:
            postings = [
                self.corpus_index.get_posting_list(token) for token in tokenized_phrase
            ]
            pointers = [0] * len(postings)
            len_before = len(candidate_docs)
            while True:
                try:
                    pointed_doc_ids = [
                        postings[idx][pointer].doc_id
                        for idx, pointer in enumerate(pointers)
                    ]
                except IndexError:
                    break
                min_idx = np.argmin(pointed_doc_ids)
                unique_pointed_docs = set(pointed_doc_ids)
                if len(unique_pointed_docs) != 1:
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
                    done = False
                    for field in FIELDS:
                        # If document contains the phrase in any fields, stop.
                        if done:
                            break
                        # Check the exact phrase by checking the positions
                        for position in ref[f"{field}_positions"]:
                            count = 0
                            for idx, other_posting_items in enumerate(
                                posting_items[1:]
                            ):
                                if (
                                    position + idx + 1
                                    in other_posting_items[f"{field}_positions"]
                                ):
                                    count += 1
                            if count == len(posting_items) - 1:
                                candidate_docs.add(doc_id)
                                done = True
                                break
                    pointers = [point + 1 for point in pointers]
            if len_before == len(candidate_docs):
                # Phrase not found
                return []

        # Remove "
        query.replace('"', " ")
        query_tokens = text_preparer.prepare_text(query)

        token_and_weights = (
            list(
                (
                    (token, 1 + np.log10(tf))
                    for token, tf in zip(
                        *list(
                            list(x) for x in np.unique(query_tokens, return_counts=True)
                        )
                    )
                )
            )
            if method == Methods.LTC_LNC
            else list((token, 1.0) for token in query_tokens)
        )

        # Initiate scores
        scores = dict()
        for field in FIELDS:
            scores[field] = dict()

        num_docs = len(self.documents)
        positive_docs = set()  # Docs with positive scores

        for query_token, token_weight in token_and_weights:
            posting_list = self.corpus_index.get_posting_list(query_token)
            if len(posting_list) == 0:
                continue
            for field in FIELDS:
                df = self.corpus_index.index[query_token].doc_frequency[field]
                if df == 0:
                    continue
                idf = np.log10(num_docs / df)
                for posting_list_item in posting_list:
                    doc_id = posting_list_item.doc_id
                    # If doing phrasal search, check for candidate docs
                    if doc_id not in candidate_docs and phrasal:
                        continue
                    tf = posting_list_item[f"{field}_tf"]
                    if tf == 0:
                        continue
                    w = 1 + np.log10(tf)
                    if doc_id not in scores[field]:
                        scores[field][doc_id] = [0.0, 0.0]
                        # Calculate weights for tokens not in query for normalization
                        for non_query_token, non_query_token_w in list(
                            (
                                (token, 1 + np.log10(tf))
                                for token, tf in zip(
                                    *list(
                                        np.unique(
                                            self.documents[doc_id][f"{field}_tokens"],
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

                    positive_docs.add(doc_id)
                    scores[field][doc_id][0] += token_weight * idf * w
                    scores[field][doc_id][1] += (idf * w) ** 2

        # Normalize scores
        normalized_scores = dict()
        for field in FIELDS:
            normalized_scores[field] = dict()
            for doc_id in scores[field]:
                normalized_scores[field][doc_id] = (
                    scores[field][doc_id][0] / np.sqrt(scores[field][doc_id][1])
                    if method == Methods.LTC_LNC
                    else scores[field][doc_id][0]
                )

        # Apply field weights
        final_scores = []
        for doc_id in positive_docs:
            score = 0
            for idx, field in enumerate(FIELDS):
                if doc_id in normalized_scores[field]:
                    score += normalized_scores[field][doc_id] * weights[idx]
            final_scores.append((doc_id, score))
        # Sort final scores
        final_scores.sort(key=lambda item: -item[1])

        relevant_docs = list(doc_id for doc_id, score in final_scores)
        return relevant_docs[:max_retrieved]

    def save_index(self, destination: str) -> None:
        with open(destination, "wb") as f:
            pickle.dump(self, f)
