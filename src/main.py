import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import untangle
from tqdm import tqdm

from src.prepare_text import TextPreparer
from src.utils import next_greater, binary_search

sys.setrecursionlimit(10 ** 6)
DocID = int
Token = str
text_preparer = TextPreparer()
FIELDS = ["title", "text"]


class Document:
    title_tokens: List[Token]
    text_tokens: List[Token]

    def __init__(self, doc_id: DocID, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.title_tokens = text_preparer.prepare_text(title)
        self.text_tokens = text_preparer.prepare_text(text)


class PostingListItem:
    def __init__(self, doc_id: DocID):
        self.doc_id = doc_id
        self.title_positions = []
        self.text_positions = []

    def add_to_positions(self, field: str, position: int):
        self.__getattribute__(f"{field}_positions").append(position)

    def __str__(self):
        return f"""
        Document ID: {self.doc_id}
        Title Positions: {self.title_positions[:10]}
        Text Positions: {self.text_positions[:10]}
        """


def create_doc(page, debug: bool = False) -> Document:
    if debug:
        print(f"Creation of document {page.id.cdata} started!")
    doc = Document(DocID(page.id.cdata), page.title.cdata, page.revision.text.cdata)
    if debug:
        print(f"Document {page.id.cdata} created!")
    return doc


def create_documents(
    docs_path: str = "./data/Persian.xml", multiprocess: bool = False
) -> List[Document]:
    tree = untangle.parse(docs_path)
    documents = []

    if multiprocess:
        pool = multiprocessing.Pool()
        for page in tree.mediawiki.page:
            documents.append(pool.apply_async(create_doc, args=(page,)))
        pool.close()
        pool.join()
        documents = [res.get() for res in documents]
    else:
        for page in tqdm(tree.mediawiki.page):
            documents.append(
                Document(
                    DocID(page.id.cdata), page.title.cdata, page.revision.text.cdata
                )
            )
    documents = sorted(documents, key=lambda document: document.doc_id)
    return documents


@dataclass
class TokenIndexItem:
    posting_list: List[PostingListItem]
    term_frequency: Dict[str, int]
    doc_frequency: Dict[str, int]


def read_document(docs_path: str, doc_id: DocID) -> Optional[Document]:
    document = read_document(docs_path, doc_id)
    return document


def create_new_token_index_item():
    term_frequency = dict()
    doc_frequency = dict()
    for field in FIELDS:
        term_frequency[field] = 0
        doc_frequency[field] = 0
    return TokenIndexItem([], term_frequency, doc_frequency)


class CorpusIndex:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus_index = self.construct_index(documents)
        self.bigram_index = self.construct_bigram_index()

    def construct_index(self, documents: List[Document]) -> Dict[str, TokenIndexItem]:
        corpus_index = dict()
        for document in tqdm(documents):
            token_positional_list_item_dict, token_frequency_dict = self.analyse_document(
                document
            )
            for token in token_positional_list_item_dict:
                if token not in corpus_index:
                    corpus_index[token] = create_new_token_index_item()
                corpus_index[token].posting_list.append(
                    token_positional_list_item_dict[token]
                )
            for field in FIELDS:
                for token in token_frequency_dict[field]:
                    corpus_index[token].term_frequency[field] += token_frequency_dict[
                        field
                    ][token]
                    corpus_index[token].doc_frequency[field] += 1

        return corpus_index

    def construct_bigram_index(self) -> Dict[str, List[str]]:
        bigram_index = dict()
        for token in self.corpus_index:
            modified_token = f"${token}$"
            for idx in range(len(modified_token) - 1):
                bigram = modified_token[idx : idx + 2]
                if bigram not in bigram_index:
                    bigram_index[bigram] = []
                bigram_index[bigram].append(token)
        return bigram_index

    def analyse_document(
        self, document: Document
    ) -> Tuple[Dict[str, PostingListItem], Dict[str, Dict[str, int]]]:
        token_positional_list_item_dict = dict()
        token_frequency_dict = dict()
        for field in FIELDS:
            token_frequency_dict[field] = dict()
            for idx, token in enumerate(document.__getattribute__(f"{field}_tokens")):
                if token not in token_positional_list_item_dict:
                    token_positional_list_item_dict[token] = PostingListItem(
                        document.doc_id
                    )
                if token not in token_frequency_dict[field]:
                    token_frequency_dict[field][token] = 0
                token_positional_list_item_dict[token].add_to_positions(field, idx)
                token_frequency_dict[field][token] += 1
        return token_positional_list_item_dict, token_frequency_dict

    def get_posting_list(self, word: str) -> List[PostingListItem]:
        return self.corpus_index[word].posting_list

    def get_words_with_bigram(self, bigram: str) -> List[Token]:
        try:
            tokens = self.bigram_index[bigram]
        except KeyError:
            return []
        return tokens

    def add_document_to_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document already exists
        if (
            binary_search(
                self.documents, Document(doc_id, "", ""), key=lambda doc: doc.doc_id
            )
            != -1
        ):
            print("Document already exists!")
            return
        # Read Document
        document = read_document(docs_path, doc_id)
        if document is None:
            print("Document not found!")
            return
        document_insertion_idx = next_greater(
            self.documents, document, key=lambda x: x.doc_id
        )
        document_insertion_idx = (
            document_insertion_idx
            if document_insertion_idx != -1
            else len(self.documents)
        )
        self.documents.insert(document_insertion_idx, document)
        token_positional_list_item_dict, token_frequency_dict = self.analyse_document(
            document
        )
        for token in token_positional_list_item_dict:
            if token not in self.corpus_index:
                self.add_token_to_index(token)
                self.corpus_index[token].posting_list.append(
                    token_positional_list_item_dict[token]
                )
            else:
                insertion_idx = next_greater(
                    self.corpus_index[token].posting_list,
                    document,
                    key=lambda x: x.doc_id,
                )
                insertion_idx = (
                    insertion_idx
                    if insertion_idx != -1
                    else len(self.corpus_index[token].posting_list)
                )
                self.corpus_index[token].posting_list.insert(
                    insertion_idx, token_positional_list_item_dict[token]
                )
        # Update doc and term frequencies
        for field in FIELDS:
            for token in token_frequency_dict[field]:
                self.corpus_index[token].term_frequency[field] += token_frequency_dict[
                    field
                ][token]
                self.corpus_index[token].doc_frequency[field] += 1
        return

    def add_token_to_index(self, token: str) -> None:
        self.corpus_index[token] = create_new_token_index_item()

    def delete_document_from_indexes(self, docs_path: str, doc_id: DocID) -> None:
        # Check if document exists
        idx = binary_search(
            self.documents, Document(doc_id, "", ""), key=lambda doc: doc.doc_id
        )
        if idx == -1:
            print("Document does not exists!")
            return
        document = read_document(docs_path, doc_id)
        if document is None:
            print("Document not found!")
            return
        del self.documents[idx]
        token_positional_list_item_dict, token_frequency_dict = self.analyse_document(
            document
        )
        for token in token_positional_list_item_dict:
            if token not in self.corpus_index:
                continue
            else:
                deletion_idx = binary_search(
                    self.corpus_index[token].posting_list,
                    document,
                    key=lambda x: x.doc_id,
                )
                if deletion_idx == -1:
                    continue
                del self.corpus_index[token].posting_list[deletion_idx]
        # Update doc and term frequencies
        for field in FIELDS:
            for token in token_frequency_dict[field]:
                self.corpus_index[token].doc_frequency[field] -= 1
                self.corpus_index[token].term_frequency[field] -= token_frequency_dict[
                    field
                ][token]
        # Delete tokens with empty posting list
        for token in token_positional_list_item_dict:
            if len(self.corpus_index[token].posting_list) == 0:
                del self.corpus_index[token]

    def save_index(self, destination: str) -> None:
        with open(destination, "wb") as f:
            pickle.dump(self, f)


def load_index(source: str) -> CorpusIndex:
    with open(source, "rb") as f:
        corpus_index = pickle.load(f)
    assert isinstance(corpus_index, CorpusIndex)
    return corpus_index


def construct_positional_indexes(
    docs_path: str = "./data/Persian.xml", multiprocess: bool = False
) -> CorpusIndex:
    documents = create_documents(docs_path, multiprocess)
    return CorpusIndex(documents)
