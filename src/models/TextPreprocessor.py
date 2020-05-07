from __future__ import unicode_literals

import re
from abc import ABC, abstractmethod
from typing import List


class TextPreprocessor(ABC):
    def __init__(
        self,
        stem: bool = True,
        lemmatize: bool = False,
        del_punctuation: bool = True,
        del_stop_words: bool = True,
    ):
        self.stem = stem
        self.del_punctuation = del_punctuation
        self.del_stop_words = del_stop_words
        self.lemmatize = lemmatize
        self.punctuation_pattern = re.compile(
            r'([؟!\?]+|\d[\d\.:\/\\]+\d|[:\.=,//،|}{؛»\]\)\}"«\[\(\{])'
        )
        super().__init__()

    def remove_punctuation(self, raw_text: str) -> str:
        return self.punctuation_pattern.sub(" ", raw_text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return list(filter(lambda token: token not in self.stop_words, tokens))

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return list(map(lambda token: self.stemmer.stem(token), tokens))

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return list(map(lambda token: self.lemmatizer.lemmatize(token), tokens))

    @abstractmethod
    def preprocess_text(self, raw_text: str) -> List[str]:
        pass

    @abstractmethod
    def tokenize_text(self, normalized_text: str) -> List[str]:
        pass


class EnglishTextPreprocessor(TextPreprocessor):
    def __init__(
        self,
        stem: bool = True,
        lemmatize: bool = False,
        del_punctuation: bool = True,
        del_stop_words: bool = True,
        stemmer: str = "porter",
    ):
        super().__init__(stem, lemmatize, del_punctuation, del_stop_words)
        if del_stop_words:
            from nltk.corpus import stopwords

            self.stop_words = set(stopwords.words("english"))
        if lemmatize:
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()
        if stem:
            if stemmer == "porter":
                from nltk.stem import PorterStemmer

                self.stemmer = PorterStemmer()
            elif stemmer == "lancaster":
                from nltk.stem import LancasterStemmer

                self.stemmer = LancasterStemmer()

    def tokenize_text(self, normalized_text: str) -> List[str]:
        from nltk.tokenize import word_tokenize

        return word_tokenize(normalized_text)

    def preprocess_text(self, raw_text: str) -> List[str]:
        if self.del_punctuation:
            raw_text = self.remove_punctuation(raw_text)
        tokens = self.tokenize_text(raw_text)
        if self.stem:
            tokens = self.stem_tokens(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return tokens


class PersianTextPreprocessor(TextPreprocessor):
    def __init__(
        self,
        stem: bool = True,
        lemmatize: bool = False,
        del_punctuation: bool = True,
        del_stop_words: bool = True,
    ):
        super().__init__(stem, lemmatize, del_punctuation, del_stop_words)
        from hazm import Normalizer, WordTokenizer, Stemmer, Lemmatizer, stopwords_list

        if del_stop_words:
            self.stop_words = set(stopwords_list())
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer(separate_emoji=True)
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()

    def preprocess_text(self, raw_text: str) -> List[str]:
        if self.del_punctuation:
            raw_text = self.remove_punctuation(raw_text)
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
