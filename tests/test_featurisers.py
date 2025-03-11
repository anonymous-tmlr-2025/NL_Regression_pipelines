import scipy
from sklearn.model_selection import train_test_split
from itertools import islice
import pytest
import pandas as pd
import numpy as np
import gensim.downloader as api
from pipeline_featurisers import (
    bag_of_words_count,
    bag_of_words_binary,
    tf_idf,
    bert_cls_features,
    bert_mean_features,
    glove_mean_features
)

@pytest.fixture
def sample_text():
    # Text8 comes already tokenized on whitespace
    dataset = pd.Series(islice(api.load("text8"), 100))
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test

@pytest.mark.parametrize("featurizer", [
    bag_of_words_count,
    bag_of_words_binary,
    tf_idf,
    bert_cls_features,
    bert_mean_features,
    glove_mean_features
])
def test_featurizer_output_shape(featurizer, sample_text):
    """
    Sanity check that the featurizer can run and returns something.
    """
    train, test = sample_text
    # Get the featurizer function
    feat_fn = featurizer(train)
    
    # Generate features
    features = feat_fn(test)
    
    # Basic shape checks
    assert features.shape[0] == len(test)
    assert features.shape[1] > 0
    
    # Check output is either numpy array or scipy sparse matrix
    is_sparse = scipy.sparse.issparse(features)
    is_array = isinstance(features, np.ndarray)
    assert is_sparse or is_array