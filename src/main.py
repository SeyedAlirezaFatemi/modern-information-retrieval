import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import untangle
from tqdm import tqdm

from src.prepare_text import TextPreparer
from src.utils import next_greater, binary_search

sys.setrecursionlimit(10 ** 6)
DocID = int

text_preparer = TextPreparer()


class Document:
    title_tokens: List[str]
    text_tokens: List[str]

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
    term_frequency: int
    doc_frequency: int


class CorpusIndex:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus_index = self.construct_index(documents)

    def construct_index(self, documents: List[Document]) -> Dict[str, TokenIndexItem]:
        corpus_index = dict()
        for document in tqdm(documents):
            token_positional_list_item_dict, token_frequency_dict = self.create_token_positional_list_item_dict(
                document
            )
            for token in token_positional_list_item_dict:
                if token not in corpus_index:
                    corpus_index[token] = TokenIndexItem([], 0, 0)
                corpus_index[token].posting_list.append(
                    token_positional_list_item_dict[token]
                )
                corpus_index[token].term_frequency += token_frequency_dict[token]
                corpus_index[token].doc_frequency += 1

        return corpus_index

    def create_token_positional_list_item_dict(
        self, document: Document
    ) -> Tuple[Dict[str, PostingListItem], Dict[str, int]]:
        token_positional_list_item_dict = dict()
        token_frequency_dict = dict()
        for field in ["title", "text"]:
            for idx, token in enumerate(document.__getattribute__(f"{field}_tokens")):
                if token not in token_positional_list_item_dict:
                    token_positional_list_item_dict[token] = PostingListItem(
                        document.doc_id
                    )
                    token_frequency_dict[token] = 0
                token_positional_list_item_dict[token].add_to_positions(field, idx)
                token_frequency_dict[token] += 1
        return token_positional_list_item_dict, token_frequency_dict

    def get_posting_list(self, word: str) -> List[PostingListItem]:
        return self.corpus_index[word].posting_list

    def add_document_to_indexes(self, docs_path: str, doc_id: DocID) -> None:
        if (
            binary_search(
                self.documents, Document(doc_id, "", ""), key=lambda doc: doc.doc_id
            )
            != -1
        ):
            print("Document already exists!")
            return
        tree = untangle.parse(docs_path)
        document = None
        for page in tree.mediawiki.page:
            if DocID(page.id.cdata) == doc_id:
                document = Document(
                    DocID(page.id.cdata), page.title.cdata, page.revision.text.cdata
                )
                break
        if document is None:
            print("Document not found!")
            return
        insertion_idx = next_greater(
            self.documents,
            document,
            key=lambda x: x.doc_id,
        )
        self.documents.insert(insertion_idx, document)
        token_positional_list_item_dict, token_frequency_dict = self.create_token_positional_list_item_dict(
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
            self.corpus_index[token].term_frequency += token_frequency_dict[token]
            self.corpus_index[token].doc_frequency += 1
        return

    def add_token_to_index(self, token: str) -> None:
        self.corpus_index[token] = TokenIndexItem([], 0, 0)

    def delete_document_from_indexes(self, docs_path: str, doc_id: DocID) -> None:
        idx = binary_search(
            self.documents, Document(doc_id, "", ""), key=lambda doc: doc.doc_id
        )
        if idx == -1:
            print("Document does not exists!")
            return
        tree = untangle.parse(docs_path)
        document = None
        for page in tree.mediawiki.page:
            if DocID(page.id.cdata) == doc_id:
                document = Document(
                    DocID(page.id.cdata), page.title.cdata, page.revision.text.cdata
                )
                break
        if document is None:
            print("Document not found!")
            return
        del self.documents[idx]
        token_positional_list_item_dict, token_frequency_dict = self.create_token_positional_list_item_dict(
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
                self.corpus_index[token].doc_frequency -= 1
                self.corpus_index[token].term_frequency -= token_frequency_dict[token]
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
