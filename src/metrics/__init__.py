from typing import List, Union

import numpy as np

from src.enums import Methods
from src.models.Manager import Manager
from src.utils.read_queries import read_queries


def r_precision(retrieved: List[int], num_relevant_docs: int) -> float:
    return np.sum(retrieved[:num_relevant_docs]) / num_relevant_docs


def precision(retrieved: List[int]) -> float:
    return np.sum(retrieved) / len(retrieved)


def recall(retrieved: List[int], num_relevant_docs: int) -> float:
    return np.sum(retrieved) / num_relevant_docs


def f_measure(retrieved: List[int], num_relevant_docs: int) -> float:
    precision_val = precision(retrieved)
    recall_val = recall(retrieved, num_relevant_docs)
    if precision_val + recall_val == 0:
        return 0
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def dcg_at_k(retrieved: List[int], k: int) -> float:
    retrieved = np.asfarray(retrieved)[:k]
    if retrieved.size:
        return np.sum(
            np.subtract(np.power(2, retrieved), 1)
            / np.log2(np.arange(2, retrieved.size + 2))
        )
    return 0.0


def ndcg_at_k(retrieved: List[int], gt: List[int], k: int) -> float:
    idcg = dcg_at_k(gt, k)
    if not idcg:
        return 0.0
    return dcg_at_k(retrieved, k) / idcg


def average_precision(retrieved: List[int]):
    results = []
    for idx, is_relevant in enumerate(retrieved):
        if is_relevant:
            results.append(sum(retrieved[: idx + 1]) / (idx + 1))
    if len(results) == 0:
        return 0
    return np.mean(results)


def evaluate_search_engine(
    manager: Manager, method: Methods, query_id: Union[str, int] = "all"
) -> None:
    queries, relevants = read_queries(query_id)
    results_r_precision = []
    results_ndcg = []
    results_f_measure = []
    results_map = []
    for query, query_relevants in zip(queries, relevants):
        num_relevant_docs = len(query_relevants)
        retrieved_docs = manager.search(query, method=method, max_retrieved=15)
        retrieved_relevance = []
        for retrieved_doc in retrieved_docs:
            if retrieved_doc in query_relevants:
                retrieved_relevance.append(1)
            else:
                retrieved_relevance.append(0)
        results_r_precision.append(r_precision(retrieved_relevance, num_relevant_docs))
        results_ndcg.append(
            ndcg_at_k(
                retrieved_relevance,
                np.ones(len(retrieved_relevance)),
                num_relevant_docs,
            )
        )
        results_f_measure.append(f_measure(retrieved_relevance, num_relevant_docs))
        results_map.append(average_precision(retrieved_relevance))
    print(
        f"""r_precision: {np.mean(results_r_precision)}
ndcg: {np.mean(results_ndcg)}
f_measure: {np.mean(results_f_measure)}
map: {np.mean(results_map)}"""
    )
