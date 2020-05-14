from typing import List, Dict

import numpy as np

from src.models import Manager
from src.models.Document import Document


class NaiveBayes:
    priors: np.ndarray
    estimations: Dict[str, np.ndarray]

    def __init__(self, manager: Manager, smoothing: int = 1, num_classes: int = 4):
        self.manager = manager
        self.smoothing = smoothing
        self.num_classes = num_classes

    def train(self):
        num_training_docs = len(self.manager.documents)
        num_class_members = np.zeros((self.num_classes,), dtype=np.int)
        num_class_tokens = np.zeros((self.num_classes,), dtype=np.int)
        # Estimate priors or P_hat (C_k)
        for doc_id, doc in self.manager.documents.items():
            num_class_members[doc.category - 1] += 1
            for field in self.manager.fields:
                num_class_tokens[doc.category - 1] += len(doc.get_tokens(field))
        self.priors = np.log10(num_class_members / num_training_docs)
        # Estimate P_hat (t_i | C_k)
        estimations = dict()
        num_total_tokens = len(self.manager.corpus_index.index)
        for token, token_item in self.manager.corpus_index.index.items():
            estimations[token] = np.zeros((self.num_classes,), dtype=np.float32)
            for posting_list_item in self.manager.corpus_index.get_posting_list(token):
                doc_category = self.manager.documents[posting_list_item.doc_id].category
                for field in self.manager.fields:
                    estimations[token][doc_category - 1] += posting_list_item.get_tf(
                        field
                    )
            estimations[token] += self.smoothing
            estimations[token] /= num_class_tokens + num_total_tokens * self.smoothing
            estimations[token] = np.log10(estimations[token])
        self.estimations = estimations

    def test(self, val_documents: List[Document]):
        num_total_tokens = len(self.manager.corpus_index.index)
        scores = np.zeros((len(val_documents), self.num_classes))
        for index, val_doc in enumerate(val_documents):
            scores[index, :] += self.priors
            for field in self.manager.fields:
                tokens = val_doc.get_tokens(field)
                for token in tokens:
                    if token in self.estimations:
                        scores[index, :] += self.estimations[token]
                    else:
                        scores[index, :] += 1 / num_total_tokens
        return np.argmax(scores, axis=1)
