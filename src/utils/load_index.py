def load_index(source: str) -> CorpusIndex:
    with open(source, "rb") as f:
        corpus_index = pickle.load(f)
    assert isinstance(corpus_index, CorpusIndex)
    return corpus_index
