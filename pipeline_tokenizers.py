"""
Goal is to follow Hugging Face's tokenizer API.

Should return a tokenizer object that has a `__call__` method
that takes in a pandas series and returns .

Cannot name file tokenizers.py because it conflicts with the Hugging Face tokenizers library.
"""

import os
import pandas as pd
import nltk
from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize import TreebankWordTokenizer
from transformers import AutoTokenizer

def make_whitespace_tokenizer():
    return lambda text: text.str.split()

def make_bpe_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return lambda text: text.apply(tokenizer.tokenize)

def make_stanford_tokenizer():
    environ_port = os.environ.get("STANFORD_CORENLP_PORT", None)
    if environ_port is None:
        environ_port = 12346
    parser = CoreNLPParser(url=f"http://localhost:{environ_port}")
    def exhaust_generator(generator):
        return lambda x: list(generator(x))
    return lambda text: text.apply(exhaust_generator(parser.tokenize))

def make_treebank_tokenizer():
    tokenizer = TreebankWordTokenizer()
    return lambda text: text.apply(tokenizer.tokenize)

def make_bert_tokenizer():
    """
    This tokenizer is compatible with non-PyTorch models.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return lambda text: text.apply(tokenizer.tokenize)

def make_sentencepiece_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("T5")
    return lambda text: text.apply(tokenizer.tokenize)
