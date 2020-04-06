from __future__ import unicode_literals

import re
from typing import List

from hazm import *


class TextPreparer:
    def __init__(self):
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer(separate_emoji=True)
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()

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

    def prepare_text(
        self,
        raw_text: str,
        stem: bool = True,
        del_punctuation: bool = False,
        lemmatize: bool = False,
        debug: bool = False,
    ) -> List[str]:
        normalized_text = self.normalize_text(raw_text)
        if debug:
            print(normalized_text)
        tokens = self.tokenize_text(normalized_text)
        if del_punctuation:
            tokens = self.remove_punctuation(tokens)
        if stem:
            tokens = self.stem_tokens(tokens)
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return tokens
