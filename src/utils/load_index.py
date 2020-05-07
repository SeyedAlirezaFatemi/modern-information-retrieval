import pickle

from src.models.Manager import Manager
from src.models.TextPreprocessor import TextPreprocessor


def load_index(source: str, text_preprocessor: TextPreprocessor) -> Manager:
    with open(source, "rb") as f:
        manager = pickle.load(f)
    assert isinstance(manager, Manager)
    manager.text_preprocessor = text_preprocessor
    return manager
