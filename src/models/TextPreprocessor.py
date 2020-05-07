from __future__ import unicode_literals

import re
from abc import ABC, abstractmethod
from typing import List


class TextPreprocessor(ABC):
    def __init__(
        self, stem: bool = True, del_punctuation: bool = True, lemmatize: bool = False
    ):
        self.stem = stem
        self.del_punctuation = del_punctuation
        self.lemmatize = lemmatize
        super().__init__()

    @abstractmethod
    def preprocess_text(self, raw_text: str) -> List[str]:
        pass


class PersianTextPreprocessor(TextPreprocessor):
    def __init__(
        self, stem: bool = True, del_punctuation: bool = True, lemmatize: bool = False
    ):
        super().__init__(stem, del_punctuation, lemmatize)
        from hazm import Normalizer, WordTokenizer, Stemmer, Lemmatizer

        self.punctuation_pattern = re.compile(
            r'([؟!\?]+|\d[\d\.:\/\\]+\d|[:\.=,//،|}{؛»\]\)\}"«\[\(\{])'
        )
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer(separate_emoji=True)
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()

    def preprocess_text(self, raw_text: str) -> List[str]:
        if self.del_punctuation:
            raw_text = self.punctuation_pattern.sub(" ", raw_text)
        normalized_text = self.normalize_text(raw_text)
        tokens = self.tokenize_text(normalized_text)
        if self.stem:
            tokens = self.stem_tokens(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return tokens

    def normalize_text(self, raw_text: str) -> str:
        return self.normalizer.normalize(raw_text)

    def tokenize_text(self, normalized_text: str) -> List[str]:
        return self.word_tokenizer.tokenize(normalized_text)

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        pattern = re.compile(r"[.!?;\\-]")
        return list(filter(lambda token: not re.match(pattern, token), tokens))

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return list(map(lambda token: self.stemmer.stem(token), tokens))

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return list(map(lambda token: self.lemmatizer.lemmatize(token), tokens))
