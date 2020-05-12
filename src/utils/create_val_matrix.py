from collections import Counter
from typing import List

import numpy as np
from tqdm import tqdm

from src.models.Document import Document
from src.models.Manager import Manager


def create_val_matrix(
    manager: Manager, val_documents: List[Document], dtype=np.float32
) -> np.ndarray:
    fields = manager.fields
    num_tokens = len(manager.corpus_index.index)
    num_train_docs = len(manager.documents)
    num_val_docs = len(val_documents)
    val_matrix = np.zeros((num_val_docs, num_tokens), dtype=dtype)
    val_documents_counts = {}
    for field in fields:
        val_documents_counts[field] = []
        for doc in val_documents:
            val_documents_counts[field].append(Counter(doc.get_tokens(field)))
    for token_index, (token, token_item) in tqdm(
        enumerate(manager.corpus_index.index.items()), total=num_tokens
    ):
        for field in fields:
            df = token_item.doc_frequency[field]
            if df == 0:
                continue
            idf = np.log10(num_train_docs / df)
            for doc_index, count in enumerate(val_documents_counts[field]):
                if token in count:
                    val_matrix[doc_index, token_index] += idf * count[token]
    return val_matrix
