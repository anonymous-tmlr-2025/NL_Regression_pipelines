import pytest
from pipeline_models import (
    get_linear_regression_model,
    get_mlp_model,
    get_resnet_model,
    get_catboost_model,
    get_xgboost_model,
)
import numpy as np
import os

@pytest.fixture
def sample_data():
    random_features = np.random.rand(100, 5)
    # Some made up convoluted target
    target = (
        random_features[:,0]*3 + 
        random_features[:,1]*random_features[:,2] + 
        random_features[:,3]**2 + 1/(random_features[:,4]+10) + 
        np.random.randn(100)*0.5
    )
    train_X, val_X = random_features[:80], random_features[80:]
    train_y, val_y = target[:80], target[80:]
    return train_X, train_y, val_X, val_y

@pytest.mark.parametrize(["get_model", "model_args"], [
    [get_linear_regression_model, {}],
    [
        get_mlp_model, 
        {
            "feature_dim": 5, 
            "hidden_sizes": [10, 10], 
            "learning_rate": 0.01, 
            "num_epochs": 1
        }
    ],
    [
        get_resnet_model, 
        {
            "feature_dim": 5, 
            "hidden_sizes": [10, 10],
            "num_subblocks": 2,
            "dropout_prob": 0.1,
            "learning_rate": 0.01,
            "num_epochs": 1
        }
    ],
    [
        get_catboost_model, 
        {"num_epochs": 1}
    ],
    [
        get_xgboost_model, 
        {
            "num_epochs": 1,
            "max_depth": 6,
            "learning_rate": 0.03,
        }
    ]
])
def test_model_output_shape(get_model, model_args, sample_data, tmp_path):
    """
    Sanity check that the model can run and returns something.
    """
    # Prevent GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Get the preprocesser function
    train_X, train_y, val_X, val_y = sample_data
    model = get_model(**model_args, output_dir=tmp_path)
    model.fit(train_X, train_y, val_X, val_y)
    preds = model.predict(val_X)
        
    # Basic shape checks
    assert preds.shape[0] == val_X.shape[0]
    assert preds.shape[1] == 1


def test_linear_regression(tmp_path):
    """
    Linear regression performs so badly that we are checking it 
    is implemented correctly.
    """
    N_samples = 100
    coefs = [1, 2, 3, 4, 5]
    X = np.random.rand(N_samples, len(coefs))
    y = 10*np.ones(N_samples).reshape(-1,1)
    for i, coef in enumerate(coefs):
        y += coef*X[:,i].reshape(-1,1)
    y += np.random.normal(0, 0.001, size=N_samples).reshape(-1,1)
    train_X, val_X = X[:80], X[80:]
    train_y, val_y = y[:80], y[80:]
    model = get_linear_regression_model(feature_dim=5, output_dir=tmp_path)
    model.fit(train_X, train_y, None, None)
    preds = model.predict(val_X)
    
    # Check that coefficients are close to true values
    assert np.allclose(model.coef_, [1, 2, 3, 4, 5], rtol=0.1)
    assert np.allclose(model.intercept_, 10, rtol=0.1)

    # Check that predictions are close to true values
    assert np.allclose(preds, val_y, rtol=0.1)
