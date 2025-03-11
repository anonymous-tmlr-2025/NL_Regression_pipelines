from functools import partial
from pathlib import Path
from typing import Annotated
from gensim.models import KeyedVectors
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import gensim.downloader as api
from scipy.sparse import csr_array, hstack
from sklearn.model_selection import train_test_split
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModel

import typer
import pandas as pd

from constants import BERT_MODEL

vectorizer_args = {
    "ngram_range": (1, 2),
    "tokenizer": lambda x: x,
    "preprocessor": lambda x: x,
    "max_features": 100000
}

def bag_of_words_count(text: pd.Series) -> csr_array:
    logger.info("Generating bag of words count featurizer")
    vectorizer = CountVectorizer(**vectorizer_args)
    vectorizer.fit(text)
    def bag_of_words_transform(text):
        logger.info("Transforming text using bag of words count featurizer")
        return vectorizer.transform(text).toarray().astype(np.float32)
    return bag_of_words_transform

def bag_of_words_binary(text: pd.Series) -> csr_array:
    logger.info("Generating bag of words binary featurizer")
    vectorizer = CountVectorizer(**vectorizer_args, binary=True)
    vectorizer.fit(text)
    def bag_of_words_transform(text):
        logger.info("Transforming text using bag of words binary featurizer")
        return vectorizer.transform(text).toarray().astype(np.float32)
    return bag_of_words_transform

def tf_idf(text: pd.Series) -> csr_array:
    logger.info("Generating tf-idf featurizer")
    vectorizer = TfidfVectorizer(**vectorizer_args)
    vectorizer.fit(text)
    def tf_idf_transform(text):
        logger.info("Transforming text using tf-idf featurizer")
        return vectorizer.transform(text).toarray().astype(np.float32)
    return tf_idf_transform

def static_bert_features(text: pd.Series, extraction_fn: callable) -> np.ndarray:
    """
    Generate BERT features from the last hidden state. Will not use any gradient
    information.
    """
    bert_model = AutoModel.from_pretrained(BERT_MODEL)
    tokeniser = AutoTokenizer.from_pretrained(BERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    text: list[list[str]] = text.tolist()
    logger.debug("Transforming text using BERT featurizer (no grad)")
    result = batch_transformer_no_grad(text, extraction_fn, bert_model, tokeniser, device)
    del bert_model, tokeniser
    return result

def batch_transformer_no_grad(
        text: list[list[str]], 
        extraction_fn: callable, 
        bert_model: AutoModel, 
        tokeniser: AutoTokenizer, 
        device: torch.device,
        batch_size: int = 100
    ) -> np.ndarray:
    """
    Apply some model using batches without any gradient information.

    Use extraction_fn to dependency inject how the model output should be turned
    into features.
    """
    with torch.no_grad():
        outputs = []
        for i in range(0, len(text), batch_size):
            batch = tokeniser(text[i:i+batch_size], is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
            batch_outputs = bert_model(**batch.to(device))
            outputs.append(extraction_fn(batch_outputs).cpu().numpy())
            del batch, batch_outputs
    return np.concatenate(outputs)

def differentiable_bert_features(
        text: pd.Series, 
        bert_model: AutoModel,
        tokeniser: AutoTokenizer,
        device: torch.device,
        extraction_fn: callable
    ) -> np.ndarray:
    """
    Needs to be passed external model so caller can control batching as well
    as optimisation of model parameters.
    """
    text:list[list[str]] = text.tolist()
    batch = tokeniser(
        text, 
        is_split_into_words=True, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    batch_outputs = bert_model(**batch.to(device))
    result = extraction_fn(batch_outputs)
    del batch, batch_outputs
    return result

def bert_cls_features(text: pd.Series, for_finetuning: bool = False) -> np.ndarray:
    """
    Generate BERT features from the cls token of the last hidden state.

    If not for_finetuning, will return a function that acts on raw text and 
    batch processes without any gradient information.

    If for_finetuning is true, will return a function that does not control 
    batching, can be differentiated by torch autograd and acts on tokenised text.
    """
    logger.info("Generating BERT cls featurizer" + (" for finetuning" if for_finetuning else ""))
    def cls_token(x):
        return x.last_hidden_state[:, 0, :]
    if for_finetuning:
        return partial(differentiable_bert_features, extraction_fn=cls_token)
    else:
        return partial(static_bert_features, extraction_fn=cls_token)

def bert_mean_features(text: pd.Series, for_finetuning: bool = False) -> np.ndarray:
    """
    Generate BERT features from the mean of the last hidden state.

    If not for_finetuning, will return a function that acts on raw text and 
    batch processes without any gradient information.

    If for_finetuning is true, will return a function that does not control 
    batching, can be differentiated by torch autograd and acts on tokenised text.
    """
    logger.info("Generating BERT mean featurizer" + (" for finetuning" if for_finetuning else ""))
    def mean_token(x):
        return x.last_hidden_state.mean(dim=1)
    if for_finetuning:
        return partial(differentiable_bert_features, extraction_fn=mean_token)
    else:
        return partial(static_bert_features, extraction_fn=mean_token)

def glove_mean_features(text: pd.Series) -> np.ndarray:
    logger.info("Generating glove mean featurizer")
    def wrap(x):
        logger.info("Transforming text using glove mean featurizer")
        embed_size = 100
        glove_vectors: KeyedVectors = api.load(f'glove-wiki-gigaword-{embed_size}')
        tokenised_text: np.array[list[str]] = x.values
        corpus_vectors = []
        for doc in tokenised_text:
            if doc:
                doc_vectors = glove_vectors.get_mean_vector(doc)
            else:
                doc_vectors = np.zeros(embed_size)
            corpus_vectors.append(doc_vectors)
        return np.stack(corpus_vectors, axis=0)
    return wrap

