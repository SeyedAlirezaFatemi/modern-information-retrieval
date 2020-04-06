from typing import List, Dict

from src.types import Token, Bigram
from .CorpusIndex import CorpusIndex


class BigramIndex:
    def __init__(self, corpus_index: CorpusIndex):
        self.index = self.construct_bigram_index(corpus_index)

    def construct_bigram_index(
        self, corpus_index: CorpusIndex
    ) -> Dict[Bigram, List[Token]]:
        bigram_index = dict()
        for token in corpus_index.index:
            modified_token = f"${token}$"
            for idx in range(len(modified_token) - 1):
                bigram = modified_token[idx : idx + 2]
                if bigram not in bigram_index:
                    bigram_index[bigram] = []
                bigram_index[bigram].append(token)
        return bigram_index

    def get_words_with_bigram(self, bigram: Bigram) -> List[Token]:
        try:
            tokens = self.index[bigram]
        except KeyError:
            print(f"Bigram {bigram} not found in index.")
            return []
        return tokens
