import pickle
from typing import List

import numpy as np

from src.enums import FIELDS, Methods
from src.models import CorpusIndex, BigramIndex
from src.models import Document
from src.types import DocID
from src.utils import next_greater, binary_search, read_document


class Manager:
    def __init__(self, documents: List[Document], text_preparer):
        self.documents = documents
        self.corpus_index = CorpusIndex(documents)
        self.bigram_index = BigramIndex(self.corpus_index)
        self.text_preparer = text_preparer

    def add_document_to_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document already exists
        # TODO: Fix
        if (
            binary_search(
                self.documents,
                Document(self.text_preparer, doc_id, "", ""),
                key=lambda doc: doc.doc_id,
            )
            != -1
        ):
            print("Document already exists!")
            return
        # Read document
        document = read_document(docs_path, doc_id, self.text_preparer)
        if document is None:
            print("Document not found!")
            return
        # Find where we must put the documents in self.documents list
        document_insertion_idx = next_greater(
            self.documents, document, key=lambda x: x.doc_id
        )
        document_insertion_idx = (
            document_insertion_idx
            if document_insertion_idx != -1
            else len(self.documents)
        )
        self.documents.insert(document_insertion_idx, document)
        self.corpus_index.add_document_to_indexes(document)

    def delete_document_from_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document exists
        # TODO: Not the best
        idx = binary_search(
            self.documents,
            Document(self.text_preparer, doc_id, "", ""),
            key=lambda doc: doc.doc_id,
        )
        if idx == -1:
            print("Document does not exists!")
            return
        document = read_document(docs_path, doc_id, self.text_preparer)
        if document is None:
            print("Document not found!")
            return
        del self.documents[idx]
        self.corpus_index.delete_document_from_indexes(document)

    def search(self, query: str, method: str = "ltn-lnn", weights=None) -> List[DocID]:
        if weights is None:
            weights = [1.0, 2.0]
        if method != Methods.LTC_LNC.value and method != Methods.LTN_LNN.value:
            print(f"Method {method} is not supported!")
            return []

        query_tokens = self.text_preparer.prepare_text(query)

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
        positive_docs = set()
        for query_token, token_weight in token_and_weights:
            posting_list = self.corpus_index.get_posting_list(query_token)
            for field in FIELDS:
                df = self.corpus_index.index[query_token].doc_frequency[field]
                try:
                    idf = np.log10(num_docs / df)
                except ZeroDivisionError:
                    continue
                for posting_list_item in posting_list:
                    doc_id = posting_list_item.doc_id
                    tf = posting_list_item[f"{field}_tf"]
                    w = 1 + np.log10(tf)
                    if np.isinf(w):
                        continue
                    if doc_id not in scores[field]:
                        scores[field][doc_id] = [0.0, 0.0]
                    positive_docs.add(doc_id)
                    scores[field][doc_id][0] += token_weight * idf * w
                    scores[field][doc_id][1] += (idf * w) ** 2

        # Normalize scores
        normalized_scores = dict()
        for field in FIELDS:
            normalized_scores[field] = dict()
            for doc_id in scores[field]:
                normalized_scores[field][doc_id] = scores[field][doc_id][0] / np.sqrt(
                    scores[field][doc_id][1]
                )

        # final_scores = dict()
        final_scores = []
        for doc_id in positive_docs:
            #     final_scores[doc_id] = 0
            score = 0
            for idx, field in enumerate(FIELDS):
                if doc_id in normalized_scores[field]:
                    #             final_scores[doc_id] += normalized_scores[field][doc_id] * weights[idx]
                    score += normalized_scores[field][doc_id] * weights[idx]
            final_scores.append((doc_id, score))

        final_scores.sort(key=lambda item: -item[1])

        relevant_docs = list(doc_id for doc_id, score in final_scores)
        return relevant_docs

    def save_index(self, destination: str) -> None:
        with open(destination, "wb") as f:
            pickle.dump(self, f)
