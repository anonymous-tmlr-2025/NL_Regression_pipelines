import pytest
import os
import numpy as np
from pipeline import get_X_y
from pipeline_tokenizers import make_bert_tokenizer
from pipeline_featurisers import bert_cls_features
from pipeline_models import get_mlp_model, get_resnet_model
from datasets import load_dataset
from constants import online_boats, SEED
import pandas as pd
from itertools import chain
import torch.nn as nn

@pytest.fixture
def sample_data():
    """
    Create sample tokenized text data for testing finetuning.
    Attempts to choose words that are close in embedding space to make it
    more challenging for the model to learn.
    """
    seed = 97
    np.random.seed(seed)
    # Take two text samples that represent concepts very close in embedding space
    close_words = [
        ("require", "requires"),
        ("fully", "easily")
        # ("years", "amateur"), # These two are very separable
    ]
    all_words = list(chain.from_iterable(close_words))
    num_batches_per_epoch = 100
    tokenised_X = np.random.choice(all_words, size=16*num_batches_per_epoch, replace=True)
    # Create target values with noise
    y = []
    for word in tokenised_X:
        if word in [x[0] for x in close_words]:
            y.append(0 + np.random.normal(0, 0.1))
        else:
            y.append(10 + np.random.normal(0, 0.1))
    y = np.array(y)
    # NOTE: Think about this; could be that the process of normalising outputs helps
    # to learn something better for finetuning.
    
    # Split into train/val sets
    train_size = 80
    tokenised_X = pd.Series([list((x,)) for x in tokenised_X], name="text")
    train_X = tokenised_X.iloc[:train_size]
    train_y = y[:train_size]
    val_X = tokenised_X.iloc[train_size:]
    val_y = y[train_size:]
    
    return train_X, train_y, val_X, val_y


def test_finetuning_save_works(sample_data, tmp_path):
    pass


def test_initial_finetune_gives_same_results(sample_data, tmp_path):
    """Test that a finetuned model initialised with the same parameters gives the same results"""
    # Prevent GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    def rmse(pred, y):
        return np.sqrt(((pred - y) ** 2).mean())
    
    train_tokens_X, train_y, val_tokens_X, val_y = sample_data
    mean_pred = np.mean(train_y)*np.ones_like(val_y)
    mean_rmse = rmse(mean_pred, val_y)

    # Create and train initial model
    model_args = {
        "feature_dim": 768,
        "hidden_sizes": [64, 64],
        "learning_rate": 0.01,
        "num_epochs": 1000,
        "output_dir": tmp_path
    }
    featuriser = bert_cls_features(train_tokens_X)
    train_features = featuriser(train_tokens_X)
    val_features = featuriser(val_tokens_X)
    model = get_mlp_model(**model_args)
    model.fit(train_features, train_y, val_features, val_y)
    # Get initial predictions
    initial_preds = model.predict(val_features)
    initial_rmse = rmse(initial_preds, val_y)

    finetune_featuriser = bert_cls_features(train_tokens_X, for_finetuning=True)
    model.finetune(
        featuriser=finetune_featuriser,
        train_X=train_tokens_X,
        train_y=train_y,
        val_X=val_tokens_X,
        val_y=val_y,
        num_epochs=0
    )
    finetuned_preds = model.predict(val_tokens_X)
    finetuned_rmse = rmse(finetuned_preds, val_y)
    
    assert np.allclose(finetuned_preds, initial_preds)
    assert np.isclose(finetuned_rmse, initial_rmse)
    