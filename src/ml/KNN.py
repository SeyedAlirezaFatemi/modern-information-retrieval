from collections import Counter
from typing import List

import numpy as np
from tqdm import tqdm

from src.enums import Measures
from src.models.Document import Document
from src.models.Manager import Manager
from src.numba_utils import fast_dot_transpose, calc_euclidean_distance
from src.utils.arg_k import arg_k
from src.utils.cosine_normalize import cosine_normalize
from src.utils.extract_train_feature_matrix import extract_train_feature_matrix
from src.utils.extract_val_feature_matrix import extract_val_feature_matrix


class KNN:
    """
    This model takes validation docs in the constructor and saves the calculated_measures in itself.
    This is because of the huge computation cost needed to calculate measures.
    """

    def __init__(
        self,
        manager: Manager,
        val_documents: List[Document],
        measure: Measures = Measures.COSINE_SIMILARITY,
    ):
        self.manager = manager
        self.val_documents = val_documents
        self.measure = measure

    def extract_feature_matrices(self):
        self.train_matrix = extract_train_feature_matrix(self.manager)
        self.val_matrix = extract_val_feature_matrix(self.manager, self.val_documents)
        if self.measure == Measures.COSINE_SIMILARITY:
            self.train_matrix = cosine_normalize(self.train_matrix)
            self.val_matrix = cosine_normalize(self.val_matrix)

    def calculate_measure(self):
        if self.measure == Measures.COSINE_SIMILARITY:
            self.calculated_measure = fast_dot_transpose(
                self.train_matrix, self.val_matrix
            )
        else:
            self.calculated_measure = calc_euclidean_distance(
                self.train_matrix, self.val_matrix
            )

    def run(self, k: int):
        ind = arg_k(
            self.calculated_measure, k, self.measure == Measures.COSINE_SIMILARITY
        )
        map_back_to_category = np.vectorize(
            lambda doc_id: self.manager.documents[doc_id].category
        )
        neighbour_categories = map_back_to_category(ind)
        predicted_categories = []
        for val_doc_index in range(neighbour_categories.shape[1]):
            counts = np.bincount(neighbour_categories[:, val_doc_index])
            predicted_categories.append(np.argmax(counts))
        return predicted_categories

    def slow_euclidean_measure_calculation(self):
        """
        calc_euclidean_distance function in numba_utils is preferred over this method.
        """
        fields = self.manager.fields
        num_train_docs = len(self.manager.documents)
        num_val_docs = len(self.val_documents)
        # Calculate token count for each field in each doc
        val_documents_counts = {}
        train_documents_counts = {}
        for field in fields:
            val_documents_counts[field] = []
            train_documents_counts[field] = []
            for doc in self.val_documents:
                val_documents_counts[field].append(Counter(doc.get_tokens(field)))
            for doc_id, doc in self.manager.documents.items():
                train_documents_counts[field].append(Counter(doc.get_tokens(field)))
        distances = np.zeros((num_val_docs, num_train_docs), dtype=np.float32)
        for field in fields:
            for val_index, val_doc_count in tqdm(
                enumerate(val_documents_counts[field]), total=num_val_docs
            ):
                for train_index, train_doc_count in enumerate(
                    train_documents_counts[field]
                ):
                    for token, token_count in val_doc_count.items():
                        if self.manager.corpus_index.get_token_item(token) is None:
                            continue
                        token_df = self.manager.corpus_index.get_token_item(
                            token
                        ).get_doc_frequency(field)
                        if token_df == 0:
                            continue
                        idf = np.log10(num_train_docs / token_df)
                        if token in train_doc_count:
                            train_token_count = train_doc_count[token]
                            distances[val_index, train_index] += (
                                (token_count - train_token_count) * idf
                            ) ** 2
                        else:
                            distances[val_index, train_index] += (
                                token_count * idf
                            ) ** 2
        self.calculated_measure = distances
