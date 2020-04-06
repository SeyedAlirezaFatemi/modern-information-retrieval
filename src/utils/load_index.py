import pickle

from src.models.Manager import Manager


def load_index(source: str) -> Manager:
    with open(source, "rb") as f:
        manager = pickle.load(f)
    assert isinstance(manager, Manager)
    return manager
