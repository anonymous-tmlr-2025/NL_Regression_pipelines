import pytest
import pandas as pd
import gensim.downloader as api
from pipeline_preprocess import (
    make_lemmatizer,
    make_stemmer,
    make_spacy_stopword_remover,
    make_nltk_stopword_remover,
    make_punctuation_remover,
    make_spacy_number_remover,
    make_simple_number_remover,
    make_lowercaser
)

@pytest.fixture
def sample_text():
    def extract_n(iterable, n, cutoff=500):
        for _ in range(n):
            item = next(iter(iterable))
            yield item[:cutoff]
    first_10 = extract_n(api.load("text8"), 10)
    dataset = pd.Series(first_10).str.join(" ")
    return dataset

@pytest.mark.parametrize("preprocesser", [
    make_lemmatizer,
    make_stemmer,
    make_spacy_stopword_remover,
    make_nltk_stopword_remover,
    make_punctuation_remover,
    make_spacy_number_remover,
    make_simple_number_remover,
    make_lowercaser
])
def test_preprocesser_output_shape(preprocesser, sample_text):
    """
    Sanity check that the preprocesser can run and returns something.
    """
    # Get the preprocesser function
    preprocesser_fn = preprocesser()
    preprocessed = preprocesser_fn(sample_text)
        
    # Basic shape checks
    assert isinstance(preprocessed, pd.Series)
    assert preprocessed.shape[0] == len(sample_text)