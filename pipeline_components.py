"""
Centralised definitions of pipeline components along with names.
"""

from pipeline_preprocess import(
    make_stemmer,
    make_lemmatizer,
    make_lowercaser,
    make_simple_number_remover,
    make_spacy_number_remover,
    make_punctuation_remover,
    make_spacy_stopword_remover,
    make_nltk_stopword_remover,
)
from pipeline_tokenizers import (
    make_stanford_tokenizer,
    make_treebank_tokenizer,
    make_whitespace_tokenizer,
    make_bpe_tokenizer,
    make_bert_tokenizer,
)
from pipeline_featurisers import (
    bag_of_words_count,
    bag_of_words_binary,
    tf_idf,
    bert_cls_features,
    bert_mean_features,
    glove_mean_features,
)
from pipeline_models import (
    get_linear_regression_model,
    get_mlp_model,
    get_relu_mlp_model,
    get_resnet_model,
    get_catboost_model,
    get_shallow_mlp_model,
    get_xgboost_model,
)
from collections import namedtuple

PipelineComponent = namedtuple("PipelineComponent", ["name", "fn"])

def filter_components_by_name(components: list[PipelineComponent], names: list[str]|str):
    if isinstance(names, str):
        return next((c for c in components if c.name == names), None)
    else:
        results = [c for c in components if c.name in names]
        if len(results) == 0 and len(names) > 0:
            raise ValueError(f"No components found with names {names}")
        return results

def iterate_component_names(names: list[PipelineComponent]):
    for component in names:
        yield component.name

# Optional preprocessing
# Maps Series (N,1) -> (N,1) where 1 is a string
preprocessers = [
    PipelineComponent("stem", make_stemmer),
    PipelineComponent("lemmatize", make_lemmatizer),
    PipelineComponent("lowercase", make_lowercaser),
    # PipelineComponent("no_numbers_simple", make_simple_number_remover),
    # PipelineComponent("no_numbers_spacy", make_spacy_number_remover),
    PipelineComponent("no_punctuation", make_punctuation_remover),
    PipelineComponent("no_stopwords_spacy", make_spacy_stopword_remover),
    PipelineComponent("no_stopwords_nltk", make_nltk_stopword_remover),
]

# Tokenisers
# Maps Series (N,1) -> (N,T) where T is the number of tokens
tokenisers = [
    PipelineComponent("whitespace", make_whitespace_tokenizer),
    PipelineComponent("bpe", make_bpe_tokenizer),
    PipelineComponent("bert", make_bert_tokenizer),
    PipelineComponent("stanford", make_stanford_tokenizer),
    PipelineComponent("treebank", make_treebank_tokenizer),
]

# Featurisers
# Maps Series (N,T) -> Numpy 3x(N,F) where F is the number of features
featurisers = [
    PipelineComponent("bow_count", bag_of_words_count),
    PipelineComponent("bow_binary", bag_of_words_binary),
    PipelineComponent("tf_idf", tf_idf),
    PipelineComponent("bert_cls", bert_cls_features),
    PipelineComponent("bert_mean", bert_mean_features),
    PipelineComponent("glove_mean", glove_mean_features),
]

# Models
# Maps Numpy array (N,F) -> Numpy array (N,1) where 1 is a prediction
# Has a fit method that takes train and val
# Has a predict method that takes test
models = [
    PipelineComponent("linear_regression", get_linear_regression_model), 
    PipelineComponent("relu_mlp", get_relu_mlp_model),
    # PipelineComponent("shallow_mlp", get_shallow_mlp_model),
    PipelineComponent("no_activation_mlp", get_mlp_model),
    PipelineComponent("resnet", get_resnet_model),
    PipelineComponent("catboost", get_catboost_model),
    PipelineComponent("xgboost", get_xgboost_model),
]
