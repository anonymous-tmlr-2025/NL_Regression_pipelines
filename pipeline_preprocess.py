"""
Preprocessers for text data.

Written in a style of building functions to reduce loading/reloading of expensive objects.
"""

from itertools import chain, islice
from pathlib import Path
import string
import sys
from typing import Annotated
import pandas as pd
import spacy
from spacy.matcher import Matcher
import nltk
import typer
import gensim.downloader as api
from loguru import logger
from tqdm import tqdm

def make_lemmatizer():
    model = spacy.load("en_core_web_lg")

    def lemmatize(text_col: pd.Series):
        logger.info("Lemmatizing with spaCy")
        values = text_col.values
        results = []
        pipe = model.pipe(values, disable=["ner", "parser", "textcat"])
        for doc in tqdm(pipe, total=len(values)):
            row = []
            for token in doc:
                if token.lemma_:
                    val = f"{token.lemma_}{token.whitespace_}"
                else:
                    val = f"{token.text}{token.whitespace_}"
                row.append(val)
            results.append("".join(row))
        return pd.Series(results, index=text_col.index)

    return lemmatize

def make_stemmer():
    stemmer = nltk.stem.SnowballStemmer("english")
    def separate_punctuation(text: str):
        left_punc = ""
        right_punc = ""
        word = text
        
        # Strip punctuation from left
        while word and word[0] in string.punctuation:
            left_punc += word[0]
            word = word[1:]
            
        # Strip punctuation from right
        while word and word[-1] in string.punctuation:
            right_punc = word[-1] + right_punc
            word = word[:-1]
            
        return (left_punc, word, right_punc)

    def rejoin_punctuation(left_punc, word, right_punc):
        return f"{left_punc}{word}{right_punc}"

    def stem(text_row):
        split_on_ws = text_row.split()
        without_punctuation = [separate_punctuation(word) for word in split_on_ws]
        stemmed = [(l, stemmer.stem(w), r ) for l, w, r in without_punctuation]
        rejoined = [rejoin_punctuation(l, w, r) for l, w, r in stemmed]
        return " ".join(rejoined)

    def stem_series(text_col: pd.Series):
        logger.info("Stemming with nltk Porter stemmer; as implemented in SnowballStemmer")
        return text_col.apply(stem)
    return stem_series

def make_nltk_stopword_remover():
    stopwords = nltk.corpus.stopwords.words("english")

    def remove_stopwords(text_col: pd.Series):
        logger.info("Removing stopwords with nltk")
        return text_col.apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))

    return remove_stopwords

def make_spacy_stopword_remover():
    model = spacy.load("en_core_web_lg")

    def remove_stopwords(text_col: pd.Series):
        logger.info("Removing stopwords with spaCy")
        matcher = Matcher(model.vocab)
        closed_classes = [
            "ADP", # Adposition
            "AUX", # Auxiliary verb 
            "CCONJ", # Coordinating conjunction
            "DET", # Determiner
            "PART", # Particle
            "PRON", # Pronoun
            "SCONJ", # Subordinating conjunction
            "PUNCT", # Punctuation
        ]
        strip_pattern = [[{"POS": {"IN": closed_classes}},]]
        matcher.add("STOP_WORDS", strip_pattern)
        filtered_texts = []
        pipe = model.pipe(text_col.values, n_process=-1)
        for i, doc in tqdm(enumerate(pipe), total=len(text_col)):
            matches = matcher(doc)
            if matches:
                pass
                # logger.debug(f"Found {len(matches)} stopwords in doc {i}")
            # TODO: If matcher has guarantees on order we can reimplement much more efficiently
            tokens_to_remove = set(chain.from_iterable(range(start, end) for _, start, end in matches))
            filtered_tokens = [t.text_with_ws for i, t in enumerate(doc) if i not in tokens_to_remove]
            filtered_texts.append("".join(filtered_tokens))
        return pd.Series(filtered_texts, index=text_col.index)

    return remove_stopwords

def make_punctuation_remover():
    def remove_punctuation(text_col: pd.Series):
        logger.info("Removing punctuation by replacing with spaces")
        def replace_punc_with_space(text: str):
            return "".join([char if char not in string.punctuation else " " for char in text])
        return text_col.apply(replace_punc_with_space)

    return remove_punctuation

def make_spacy_number_remover():
    model = spacy.load("en_core_web_lg")

    def remove_numbers(text_col: pd.Series):
        logger.info("Removing numbers by replacing with spaces")
        filtered_texts = []
        for doc in model.pipe(text_col.values):
            filtered_tokens = [t.text_with_ws for t in doc if not t.like_num]
            filtered_texts.append("".join(filtered_tokens))
        return pd.Series(filtered_texts, index=text_col.index)

    return remove_numbers

def make_simple_number_remover():
    def remove_numbers(text_col: pd.Series):
        logger.info("Removing numbers by replacing with spaces")
        return text_col.str.replace(r"\d+", " ", regex=True)
    return remove_numbers

def make_lowercaser():
    def lowercaser(text_col: pd.Series):
        logger.info("Lowercasing text")
        return text_col.str.lower()
    return lowercaser


app = typer.Typer()

PREPROCESSERS = {
    "lemmatizer": make_lemmatizer,
    "stemmer": make_stemmer,
    "spacy_stopword_remover": make_spacy_stopword_remover,
    "nltk_stopword_remover": make_nltk_stopword_remover,
    "punctuation_remover": make_punctuation_remover,
    "spacy_number_remover": make_spacy_number_remover,
    "simple_number_remover": make_simple_number_remover,
    "lowercaser": make_lowercaser,
}

@app.command()
def main(
    preprocesser: Annotated[str, typer.Option(help="Preprocesser to use")],
    input_file: Annotated[Path | None, typer.Option(help="Input file to preprocess")] = None,
):
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.info("Starting preprocessing pipeline")
    if preprocesser not in PREPROCESSERS:
        raise ValueError(f"Invalid preprocesser: {preprocesser}. Must be one of {list(PREPROCESSERS.keys())}")
    _preprocesser = PREPROCESSERS[preprocesser]
    logger.info(f"Using preprocesser: {preprocesser}")
    if input_file is None:
        input_file = Path("text8")
        gensim_data = islice(api.load("text8"), 100)
        sample_data = pd.Series(gensim_data).str.join(" ")
    else:
        sample_data = pd.read_csv(input_file)
    preprocess_fn = _preprocesser()
    processed = preprocess_fn(sample_data)
    processed.to_csv(f"{preprocesser}_{input_file.stem}.csv", index=False)

if __name__ == "__main__":
    app()