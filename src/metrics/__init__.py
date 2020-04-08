from typing import List, Union

import numpy as np

from src.enums import Methods
from src.models.Manager import Manager
from src.utils.read_queries import read_queries


def r_precision(relevant: List[int], num_relevant_docs: int) -> float:
    return np.sum(relevant[:num_relevant_docs]) / num_relevant_docs


def precision(relevant: List[int]) -> float:
    return np.sum(relevant) / len(relevant)


def recall(relevant: List[int], num_relevant_docs: int) -> float:
    return np.sum(relevant) / num_relevant_docs


def f_measure(relevant: List[int], num_relevant_docs: int) -> float:
    precision_val = precision(relevant)
    recall_val = recall(relevant, num_relevant_docs)
    if precision_val + recall_val == 0:
        return 0
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
    if len(results) == 0:
        return 0
    return np.mean(results)


def evaluate_search_engine(
    manager: Manager, method: Methods, query_id: Union[str, int] = "all"
):
    queries, relevants = read_queries(query_id)
    results_r_precision = []
    results_ndcg = []
    results_f_measure = []
    results_map = []
    for query, query_relevants in zip(queries, relevants):
        num_relevant_docs = len(query_relevants)
        retrieved_docs = manager.search(
            query, method=method, max_retrieved=num_relevant_docs
        )
        relevance = []
        for retrieved_doc in retrieved_docs:
            if retrieved_doc in query_relevants:
                relevance.append(1)
            else:
                relevance.append(0)
        results_r_precision.append(r_precision(relevance, num_relevant_docs))
        results_ndcg.append(ndcg_at_k(relevance, num_relevant_docs))
        results_f_measure.append(f_measure(relevance, num_relevant_docs))
        results_map.append(average_precision(relevance))
    return (
        np.mean(results_r_precision),
        np.mean(results_ndcg),
        np.mean(results_f_measure),
        np.mean(results_map),
    )
