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
