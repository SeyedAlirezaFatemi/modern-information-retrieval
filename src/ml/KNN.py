from typing import List

import numpy as np

from src.enums import Measures
from src.models.Document import Document
from src.models.Manager import Manager
from src.numba_utils import calc_similarities, calc_euclidean_distance
from src.utils.arg_k import arg_k
from src.utils.cosine_normalize import cosine_normalize
from src.utils.create_train_matrix import create_train_matrix
from src.utils.create_val_matrix import create_val_matrix


class KNN:
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
        self.train_matrix = create_train_matrix(self.manager)
        self.val_matrix = create_val_matrix(self.manager, self.val_documents)
        if self.measure == Measures.COSINE_SIMILARITY:
            self.train_matrix = cosine_normalize(self.train_matrix)
            self.val_matrix = cosine_normalize(self.val_matrix)

    def calculate_measure(self):
        if self.measure == Measures.COSINE_SIMILARITY:
            self.calculated_measure = calc_similarities(
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
