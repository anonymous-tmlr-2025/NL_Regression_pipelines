import pytest
import os
import numpy as np
from pipeline import get_X_y
from pipeline_tokenizers import make_bert_tokenizer
from pipeline_featurisers import bert_cls_features
from pipeline_models import get_mlp_model, get_relu_mlp_model
from datasets import load_dataset
from constants import online_boats, SEED


@pytest.fixture
def real_online_boats_data():
    train, val, _= load_dataset(online_boats, seed=SEED)
    tX, ty = get_X_y(train)
    vX, vy = get_X_y(val)
    bert_tokeniser = make_bert_tokenizer()
    tX = bert_tokeniser(tX)
    vX = bert_tokeniser(vX)
    return tX, ty, vX, vy


@pytest.mark.parametrize("finetuning_factor", [1, 10, 100, 1000])

def test_finetune_improves_performance(real_online_boats_data, tmp_path, finetuning_factor):
    # TODO: Rerun MLP/ResNet with smaller finetuning factor
    # TODO: Rerun MLP with activation function
    # TODO: Rerun MLP with no hidden layers to emulate linear regression
    # Prevent GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    def rmse(pred, y):
        return np.sqrt(((pred - y) ** 2).mean())
    
    train_tokens_X, train_y, val_tokens_X, val_y = real_online_boats_data
    mean_pred = np.mean(train_y)*np.ones_like(val_y)
    mean_rmse = rmse(mean_pred, val_y)

    # Create and train initial model
    model_args = {
        "feature_dim": 768,
        "output_dir": tmp_path,
        "hidden_sizes": [64, 64],
        "learning_rate": 0.01,
        "num_epochs": 1000,
        "finetuning_factor": finetuning_factor,
    }
    featuriser = bert_cls_features(train_tokens_X)
    train_features = featuriser(train_tokens_X)
    val_features = featuriser(val_tokens_X)
    trainer = get_relu_mlp_model(**model_args)
    trainer.fit(train_features, train_y, val_features, val_y)
    # Get initial predictions
    initial_preds = trainer.predict(val_features)
    initial_rmse = rmse(initial_preds, val_y)

    finetune_featuriser = bert_cls_features(train_tokens_X, for_finetuning=True)
    trainer.finetune(
        featuriser=finetune_featuriser,
        train_X=train_tokens_X,
        train_y=train_y,
        val_X=val_tokens_X,
        val_y=val_y,
        num_epochs=1000
    )
    finetuned_preds = trainer.predict(val_tokens_X)
    finetuned_rmse = rmse(finetuned_preds, val_y)

    assert finetuned_rmse < initial_rmse