from __future__ import unicode_literals

import re
import sys
from typing import List

from hazm import *

sys.setrecursionlimit(10 ** 6)

normalizer = Normalizer()
word_tokenizer = WordTokenizer(separate_emoji=True)
stemmer = Stemmer()


def normalize_text(raw_text: str) -> str:
    return normalizer.normalize(raw_text)


def tokenize_text(normalized_text: str) -> List[str]:
    return word_tokenizer.tokenize(normalized_text)


def remove_punctuation(tokens: List[str]) -> List[str]:
    pattern = re.compile(r"[.!?;\\-]")
    return list(filter(lambda token: not re.match(pattern, token), tokens))


def stem_tokens(tokens: List[str]) -> List[str]:
    return list(map(lambda token: stemmer.stem(token), tokens))


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    lemmatizer = Lemmatizer()
    return list(map(lambda token: lemmatizer.lemmatize(token), tokens))


def prepare_text(
    raw_text: str,
    stem: bool = True,
    del_punctuation: bool = False,
    lemmatize: bool = False,
    debug: bool = False,
) -> List[str]:
    normalized_text = normalize_text(raw_text)
    if debug:
        print(normalized_text)
    tokens = tokenize_text(normalized_text)
    if del_punctuation:
        tokens = remove_punctuation(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    return tokens
