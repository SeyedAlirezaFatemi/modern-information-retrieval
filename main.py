from src.enums import *
from src.metrics import evaluate_search_engine
from src.utils.load_index import load_index
from src.utils.construct_positional_indexes import construct_positional_indexes
from src.utils.load_index import load_index
from src.models.TextPreprocessor import PersianTextPreprocessor
from src.enums import *

if __name__ == "__main__":
    manager = construct_positional_indexes()
    manager.save_index("manager.pickle")

    manager = load_index("manager.pickle")
    # manager.add_document_to_indexes("./data/Add.xml", 7157)
    evaluate_search_engine(manager, method=Methods.LTN_LNN)
    # manager.search({Fields.TITLE: '"آقا علیرضا"', Fields.TEXT: "علیرضا"})
