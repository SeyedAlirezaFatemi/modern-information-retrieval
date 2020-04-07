from typing import List

import numpy as np


def r_precision(relevant: List[int], num_relevant_docs: int) -> float:
    return np.sum(relevant[:num_relevant_docs]) / num_relevant_docs


def precision(relevant: List[int]) -> float:
    return np.sum(relevant) / len(relevant)


def recall(relevant: List[int], num_relevant_docs: int) -> float:
    return np.sum(relevant) / num_relevant_docs


def f_measure(relevant: List[int], num_relevant_docs: int) -> float:
    precision_val = precision(relevant)
    recall_val = recall(relevant, num_relevant_docs)
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(
            np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2))
        )
    return 0.0


def ndcg_at_k(r, k):
    idcg = dcg_at_k(np.ones(len(r)), k)
    if not idcg:
        return 0.0
    return dcg_at_k(r, k) / idcg


def average_precision(relevant: List[int]):
    results = []
    for idx, is_relevant in enumerate(relevant):
        if is_relevant:
            results.append(sum(relevant[: idx + 1]) / (idx + 1))
    return np.mean(results)
