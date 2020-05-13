import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


class MLMetrics:
    """
    https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    https://www.python-course.eu/confusion_matrix.php
    """

    @staticmethod
    def precision(label: int, confusion_matrix: np.ndarray) -> float:
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    @staticmethod
    def recall(label: int, confusion_matrix: np.ndarray) -> float:
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    @staticmethod
    def accuracy(confusion_matrix: np.ndarray) -> float:
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    @staticmethod
    def compute_confusion_matrix(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Computes a confusion matrix using numpy for two np.arrays
        true and pred.

        Results are identical (and similar in computation time) to:
          "from sklearn.metrics import confusion_matrix"

        However, this function avoids the dependency on sklearn."""

        num_classes = len(np.unique(true))
        result = np.zeros((num_classes, num_classes))

        for i in range(len(true)):
            result[true[i]][pred[i]] += 1

        return result

    @staticmethod
    def precision_macro_average(confusion_matrix: np.ndarray) -> float:
        rows, columns = confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += MLMetrics.precision(label, confusion_matrix)
        return sum_of_precisions / rows

    @staticmethod
    def recall_macro_average(confusion_matrix: np.ndarray) -> float:
        rows, columns = confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += MLMetrics.recall(label, confusion_matrix)
        return sum_of_recalls / columns

    @staticmethod
    def f1_macro_average(confusion_matrix: np.ndarray, beta: float = 1.0) -> float:
        recall_m = MLMetrics.recall_macro_average(confusion_matrix)
        precision_m = MLMetrics.precision_macro_average(confusion_matrix)
        return ((beta ** 2 + 1) * precision_m * recall_m) / (
            (beta ** 2) * precision_m + recall_m
        )

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray) -> None:
        sn.heatmap(confusion_matrix, annot=True)
        plt.show()
