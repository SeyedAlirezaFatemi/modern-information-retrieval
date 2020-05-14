import numpy as np
from tqdm import tqdm

from src.models.Manager import Manager


def extract_train_feature_matrix(manager: Manager, dtype=np.float32) -> np.ndarray:
    """
    Note: idf is calculated for each field individually bases on the doc_frequency
        of the token in the specific field.
    """
    fields = manager.fields
    num_tokens = len(manager.corpus_index.index)
    num_train_docs = len(manager.documents)
    train_feature_matrix = np.zeros((num_train_docs, num_tokens), dtype=dtype)
    for index, (token, token_item) in tqdm(
        enumerate(manager.corpus_index.index.items()), total=num_tokens
    ):
        for field in fields:
            df = token_item.get_doc_frequency(field)
            if df == 0:
                continue
            idf = np.log10(num_train_docs / df)
            for posting_list_item in token_item.posting_list:
                doc_id = posting_list_item.doc_id
                tf = posting_list_item.get_tf(field)
                train_feature_matrix[doc_id, index] += tf * idf
    return train_feature_matrix
