from typing import Dict, Set

from src.types import Token, Bigram
from .CorpusIndex import CorpusIndex


class BigramIndex:
    def __init__(self, corpus_index: CorpusIndex):
        self.index = self.construct_bigram_index(corpus_index)

    def construct_bigram_index(
        self, corpus_index: CorpusIndex
    ) -> Dict[Bigram, Set[Token]]:
        bigram_index = dict()
        for token in corpus_index.index:
            modified_token = f"${token}$"
            for idx in range(len(modified_token) - 1):
                bigram = modified_token[idx : idx + 2]
                if bigram not in bigram_index:
                    bigram_index[bigram] = set()
                bigram_index[bigram].add(token)
        return bigram_index

    def get_words_with_bigram(self, bigram: Bigram) -> Set[Token]:
        try:
            tokens = self.index[bigram]
        except KeyError:
            print(f"Bigram {bigram} not found in index.")
            return set()
        return tokens
