"""
Models for the pipeline
"""

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time
from typing import Optional
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, record_function
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import random
from constants import SEED, BERT_MODEL
from transformers import AutoModel, AutoTokenizer

def seed_everything():
    """
    Seed everything that could be a source of randomness
    """
    # Python random
    random.seed(SEED)

    # Numpy
    np.random.seed(SEED)

    # Torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

class ResnetBlock(nn.Module):
    def __init__(self, hidden_dim, num_subblocks, dropout_prob):
        super().__init__()
        self.model = nn.Sequential(
            *[self.get_subblock(hidden_dim, dropout_prob) for _ in range(num_subblocks)],
        )
    
    def forward(self, x):
        return self.model(x) + x
    
    def get_subblock(self, hidden_dim, dropout_prob):
        return nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim)
        )

class MeanTrainer:  
    def fit(self, train_X, train_y, val_X, val_y):
        self.pred = np.concatenate([train_y, val_y], axis=0).mean()
    
    def predict(self, test_X):
        return np.full(test_X.shape[0], self.pred)
    
    def save(self, output_dir: Path):
        with open(output_dir / "mean.json", "w") as f:
            json.dump({"pred": self.pred}, f)

class LinearRegressionTrainer(SklearnLinearRegression):
    def fit(self, train_X, train_y, val_X, val_y, *args, **kwargs):
        super().fit(train_X, train_y)

    def predict(self, test_X):
        preds = super().predict(test_X)
        return preds.reshape(-1, 1)
    
    def save(self, output_dir: Path):
        params = self.get_params()
        params["coef_"] = self.coef_.tolist()
        params["intercept_"] = self.intercept_.tolist()
        with open(output_dir / "linear_regression.json", "w") as f:
            json.dump(params, f)

def torch_train(
        model,
        optimizer,
        train_X, 
        train_y, 
        val_X, 
        val_y,
        device,
        num_epochs,
        writer,
        early_stopping,
        report_every,
        time_limit,
        output_dir,
        featurised,
        track_memory=False,
        low_disk_mode=False,
    ):
    """
    Train a torch model for given number of epochs on a given device, using
    early stopping and shuffling data before training.

    Will stop training if time_limit (seconds) is reached.

    Will change state of model and optimizer.

    `writer` should be a tensorboard writer - will write _every_ epoch.

    `report_every` is the number of epochs between logging messages.

    `output_dir` should be a valid, existing directory that will not conflict
    with any other experiment - where we can safely write artefacts to.

    `featurised` should be True if the model expects featurised data (e.g. numpy
    array of floats). False if the model expects tokenised data (e.g. Series of
    lists of tokens). Will affect batch size used as we assume that model that 
    takes in tokenized text will need more memory per example.

    `track_memory` will record memory usage if true.
    """
    if featurised:
        batch_size = 32
    else:
        batch_size = 16

    # This may not match a complete batch size as it is a running average
    # across bathes
    best_val_loss = torch_evaluate(
        model, 
        val_X, 
        val_y, 
        device, 
        featurised=featurised,
        batch_size=batch_size if not featurised else None,
    )
    logger.debug(f"Begin training with initial val loss: {best_val_loss}")
    # Early stopping might be suffering from quantisation issues
    best_model = deepcopy(model.state_dict())
    epochs_no_improve = 0

    start = time.time()
    for i in range(num_epochs):
        with record_function("torch_train_epoch") if track_memory else nullcontext():
            train_loss = torch_train_epoch(
                model, 
                optimizer, 
                train_X, 
                train_y, 
                device,
                batch_size=batch_size,
                featurised=featurised,
            )
        writer.add_scalar("train_loss", train_loss, i)

        with record_function("torch_evaluate") if track_memory else nullcontext():
            val_loss = torch_evaluate(
                model, 
                val_X, 
                val_y, 
                device, 
                featurised=featurised,
                batch_size=batch_size if not featurised else None,
            )
        writer.add_scalar("val_loss", val_loss, i)

        if track_memory:
            logger.warning(f"Dumping memory snapshot at epoch {i}")
            torch.cuda.memory._dump_snapshot("train_memory_snapshot.pickle")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            if not low_disk_mode:
                torch.save(best_model, output_dir / "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping:
            logger.info(f"Early stopping at epoch {i}, loss {val_loss}; resetting model to best model which had val loss {best_val_loss}")
            model.load_state_dict(best_model)
            break

        # Logging
        if i % report_every == 0:
            logger.info(f"Epoch {i}; train loss {train_loss}; val loss {val_loss}")
        
        # Time limit
        if time.time() - start > time_limit:
            logger.info(f"Time limit reached; stopping at epoch {i} - restoring best model")
            model.load_state_dict(best_model)
            break

def torch_train_epoch(
        model, 
        optimizer,
        train_X, 
        train_y, 
        device,
        batch_size=32,
        featurised=True,
        verbose=False,
    ):
    """
    Train the model for one epoch, use early stopping and shuffle data
    before training

    Will change state of model and optimizer.
    """
    model.train()
    num_samples = train_X.shape[0]
    indices = np.random.permutation(num_samples)
    train_X, train_y = train_X[indices], train_y[indices]
    epoch_loss = 0
    optimizer.zero_grad()
    for i in range(0, num_samples, batch_size):
        batch_X, batch_y = train_X[i:i+batch_size], train_y[i:i+batch_size]
        if featurised:
            batch_X = torch.from_numpy(batch_X).float().to(device)
        batch_y = torch.from_numpy(batch_y).float().to(device)
        if batch_X.shape[0] == 1:
            logger.warning(f"Only one sample in batch {i} - skipping to avoid issues with BatchNorm")
            continue
        pred = model(batch_X)
        loss = F.mse_loss(pred, batch_y)
        loss.backward()
        epoch_loss = (epoch_loss * i)/(i+1) + loss.item()/(i+1)
        optimizer.step()
        optimizer.zero_grad()
        del batch_X, batch_y, pred, loss
        if verbose and device == "cuda":
            log_memory_usage(device)
    return epoch_loss

def log_memory_usage(device, tag: str | None=None):
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
    message = f"GPU memory allocated: {memory_allocated:.2f}MB, reserved: {memory_reserved:.2f}MB"
    if tag is not None:
        message = f"[{tag}] " + message
    logger.debug(message)

def torch_evaluate(
        model, 
        X, 
        y,
        device,
        eval_fn=None,
        batch_size=None,
        featurised=True,
    ):
    """
    Evaluate the model on a given dataset using a given evaluation function.

    If no batch_size is provided, will use the entire dataset as a single batch.

    If no eval_fn is provided, will use mean squared error. Should be compatible
    with any eval_fn that takes in two np.ndarrays and returns a scalar.

    If featurised is true, will expect featurised data (e.g. numpy array of floats).
    Otherwise will expect tokenised data (e.g. Series of lists of tokens) - 
    to be handled by the model.
    """
    if eval_fn is None:
        eval_fn = lambda preds, y: np.mean((preds - y)**2)
    if batch_size is None:
        if featurised:
            batch_size = 32
        else:
            batch_size = 16
    preds = torch_predict(model, X, device, batch_size, featurised)
    return eval_fn(preds, y)

def torch_predict(
        model,
        X,
        device,
        batch_size=None,
        featurised=True,
    ):
    """
    Predict on a given dataset.

    If no batch_size is provided, will use the entire dataset as a single batch.

    If featurised is true, will expect featurised data (e.g. numpy array of floats).
    Otherwise will expect tokenised data (e.g. Series of lists of tokens) - 
    to be handled by the model.
    """
    model.eval()
    num_samples = X.shape[0]
    if batch_size is None:
        batch_size = num_samples
    preds = []
    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i+batch_size]
        if featurised:
            batch_X = torch.from_numpy(batch_X).float().to(device)
        with torch.no_grad():
            pred = model(batch_X)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds)

class TorchModelWithBERTFeaturiser(nn.Module):
    def __init__(self, model, featuriser, device):
        super().__init__()
        self.model = model
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.tokeniser = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.device = device
        self.differentiable_featuriser = featuriser
        self.bert.to(self.device)
        self.model.to(self.device)

    def forward(self, x):
        embedding = self.differentiable_featuriser(
            x, 
            bert_model=self.bert,
            tokeniser=self.tokeniser, 
            device=self.device,
        )
        return self.model(embedding)

class TorchTrainer:
    def __init__(
            self, 
            model,
            optimizer_gen,
            output_dir: Path,
            num_epochs=10000,
            early_stopping=10,
            time_limit=3600*2,
            finetune_opt_gen=None,
        ):
        """
        Manager for training a torch model. Also handles finetuning.

        Uses same num_epochs, early_stopping for training and finetuning.

        Shares time_limit between training and finetuning in total.

        Args:
            model: torch model
            optimizer_gen: optimizer generator - should take only model parameters
            writer: tensorboard writer
            num_epochs: number of epochs to train
            early_stopping: early stopping patience in number of epochs
            time_limit: time limit in seconds
            finetune_opt_gen: optimizer generator for finetuning - should take only model parameters
        """
        super().__init__()
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer_gen = optimizer_gen
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.epochs_no_improve = 0
        self.best_val_loss = np.inf
        self.best_model = None
        self.writer = SummaryWriter(output_dir)

        # Random nice-to-have
        self.report_every = 20
        self.time_limit = time_limit
        self.time_spent = 0
        self.output_dir = output_dir
        self.trained = False
        self.finetuned = False
        self.finetuning_optimizer = finetune_opt_gen
        self.finetune_writer = SummaryWriter(output_dir, comment="finetune")
    
    def fit(self, train_X: np.ndarray, train_y, val_X: np.ndarray, val_y, low_disk_mode=False):
        logger.info(f"Training for {self.num_epochs} epochs with early stopping {self.early_stopping}")
        optimizer_copy = self.optimizer_gen(self.model.parameters())
        start = time.time()
        torch_train(
            model=self.model,
            optimizer=optimizer_copy,
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            device=self.device,
            num_epochs=self.num_epochs,
            writer=self.writer,
            early_stopping=self.early_stopping,
            report_every=self.report_every,
            time_limit=self.time_limit,
            output_dir=self.output_dir,
            featurised=True,
            low_disk_mode=low_disk_mode,
        )
        self.time_spent = time.time() - start
        self.trained = True
    
    def predict(self, test_X):
        if self.finetuned:
            logger.info("Predicting with finetuned model")
            return torch_predict(self.finetuned_model, test_X, self.device, featurised=False, batch_size=16)
        else:
            logger.info("Predicting with non-finetuned model")
            return torch_predict(self.model, test_X, self.device, featurised=True)

    def save(self, output_dir: Path):
        if self.finetuned:
            logger.info("Saving finetuned model")
            torch.save(self.finetuned_model.state_dict(), output_dir / "finetuned_model.pth")
        else:
            logger.info("Saving non-finetuned model")
            torch.save(self.model.state_dict(), output_dir / "torch_model.pth")

    def load(self, path: Path, finetuned: bool = False, featuriser: Optional[callable] = None):
        if finetuned:
            if featuriser is None:
                raise ValueError("No featuriser provided for intialising but attempting to load finetuned model")
            self.finetuned = True
            self.finetuned_model = TorchModelWithBERTFeaturiser(
                model=self.model, 
                featuriser=featuriser,
                device=self.device,
            )
            self.finetuned_model.load_state_dict(torch.load(path))
            self.model = self.finetuned_model.model
        else:
            self.model.load_state_dict(torch.load(path))

    def finetune(
            self, 
            featuriser, 
            train_X: pd.Series, 
            train_y, 
            val_X: pd.Series, 
            val_y,
            track_memory=False,
            num_epochs=None,
            low_disk_mode=False,
        ):
        if not self.trained:
            raise ValueError("Attempting to finetune before training")
        
        # Need to include the featuriser in the model
        model_with_featuriser = TorchModelWithBERTFeaturiser(
            model=self.model, 
            featuriser=featuriser,
            device=self.device,
        )
        logger.info(f"Finetuning for {self.num_epochs} epochs with early stopping {self.early_stopping}")
        
        if self.finetuning_optimizer is None:
            raise ValueError("No finetuning optimizer provided but attempting to finetune")
        optimizer_copy = self.finetuning_optimizer(model_with_featuriser.parameters())
        
        if track_memory:
            logger.warning("Recording memory history")
            torch.cuda.memory._record_memory_history(max_entries=100000)
            manager = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            logger.warning("Due to memory recording, finetuning will only run for 2 epochs")
        else:
            manager = nullcontext()
        
        with manager as prof:
            torch_train(
                model=model_with_featuriser,
                optimizer=optimizer_copy,
                train_X=train_X.values, # Convert to numpy array to allow indexing
                train_y=train_y,
                val_X=val_X.values, # Convert to numpy array to allow indexing
                val_y=val_y,
                device=self.device,
                num_epochs=self.num_epochs if num_epochs is None else num_epochs,
                writer=self.finetune_writer,
                early_stopping=self.early_stopping,
                report_every=self.report_every,
                time_limit=self.time_limit-self.time_spent,
                output_dir=self.output_dir,
                featurised=False,
                track_memory=track_memory,
                low_disk_mode=low_disk_mode,
            )
        self.finetuned_model = model_with_featuriser
        self.finetuned = True
        
        if track_memory:
            logger.warning("Disabling memory recording")
            torch.cuda.memory._record_memory_history(enabled=None)
            try:
                logger.info("Exporting memory timeline")
                prof.export_memory_timeline("memory_timeline.html")
            except Exception:
                logger.warning(f"Failed to save memory info")
        

class CatBoostTrainer:
    def __init__(
            self, 
            output_dir: Path,
            depth: int = 6,
            learning_rate: float = 0.03,
            patience: int = 10,
            num_epochs: int = 1000,
        ):
        self.output_dir = output_dir
        self.patience = patience
        self.model = CatBoostRegressor(
            iterations=num_epochs,
            loss_function="RMSE",
            early_stopping_rounds=patience,
            depth=depth,
            learning_rate=learning_rate,
            train_dir=output_dir,
            random_seed=SEED,
        )

    def fit(self, train_X, train_y, val_X, val_y, *args, **kwargs):
        return self.model.fit(train_X, train_y, eval_set=(val_X, val_y))
    
    def predict(self, test_X):
        preds = self.model.predict(test_X)
        return preds.reshape(-1, 1)

    def save(self, output_dir: Path):
        self.model.save_model(output_dir / "catboost.json")

class XGBoostTrainer:
    def __init__(
            self,
            output_dir: Path,
            num_epochs: int = 1000,
            early_stopping: int = 10,
            max_depth: int = 6,
            learning_rate: float = 0.03,
            reg_gamma: Optional[float] = None,
            reg_alpha: Optional[float] = None,
            reg_lambda: Optional[float] = None,
        ):
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=num_epochs,
            gamma=reg_gamma,
            early_stopping_rounds=early_stopping,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=SEED,
        )

    # TODO: Hacking to allow for arbitrary arguments without error, but dont use them
    def fit(self, train_X, train_y, val_X, val_y, *args, **kwargs):
        result = self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)])
        return result
    
    def predict(self, test_X):
        preds = self.model.predict(test_X)
        return preds.reshape(-1, 1)
    
    def save(self, output_dir: Path):
        self.model.save_model(output_dir / "xgboost.json")

def get_linear_regression_model(
        feature_dim: int,
        output_dir: Path,
    ):
    return LinearRegressionTrainer()

def get_mlp_model(
        feature_dim: int,
        output_dir: Path,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        num_epochs: int = 1000,
        finetuning_factor: float = 100,
        activation: nn.Module | None = None,
    ):
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    
    hidden_layers = []
    if activation is not None:
        logger.debug(f"Using activation {activation}")
    else:
        logger.debug("No activation function provided for MLP")
    for i in range(len(hidden_sizes)-1):
        hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        if activation is not None:
            hidden_layers.append(activation())
    model = nn.Sequential(
        nn.Linear(feature_dim, hidden_sizes[0]),
        *hidden_layers,
        nn.Linear(hidden_sizes[-1], 1),
    )
    optimizer = partial(torch.optim.Adam, lr=learning_rate)
    logger.debug(f"Using learning rate {learning_rate} for training")
    finetuning_optimizer = partial(torch.optim.Adam, lr=learning_rate/finetuning_factor)
    logger.debug(f"Using learning rate {finetuning_factor} times smaller for finetuning")
    
    return TorchTrainer(
        model, 
        optimizer, 
        output_dir, 
        num_epochs=num_epochs,
        finetune_opt_gen=finetuning_optimizer,
    )

def get_relu_mlp_model(
        feature_dim: int,
        output_dir: Path,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        num_epochs: int = 1000,
        finetuning_factor: float = 100,
    ):
    return get_mlp_model(feature_dim, output_dir, hidden_sizes, learning_rate, num_epochs, finetuning_factor, nn.ReLU)

def get_shallow_mlp_model(
        feature_dim: int,
        output_dir: Path,
        learning_rate: float = 0.001,
        num_epochs: int = 1000,
        finetuning_factor: float = 100,
    ):
    model = nn.Sequential(
        nn.Linear(feature_dim, 1),
    )
    optimizer = partial(torch.optim.Adam, lr=learning_rate)
    logger.debug(f"Using learning rate {learning_rate} for training")
    finetuning_optimizer = partial(torch.optim.Adam, lr=learning_rate/finetuning_factor)
    logger.debug(f"Using learning rate {finetuning_factor} times smaller for finetuning")
    return TorchTrainer(
        model, 
        optimizer, 
        output_dir, 
        num_epochs=num_epochs,
        finetune_opt_gen=finetuning_optimizer,
    )

def get_resnet_model(
        feature_dim: int,
        output_dir: Path,
        hidden_sizes: list[int] | None = None,
        num_subblocks: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 0.001,
        num_epochs: int = 1000,
        finetuning_factor: float = 100,
    ):
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    
    model = nn.Sequential(
        nn.Linear(feature_dim, hidden_sizes[0]),
        *[ResnetBlock(hidden_size, num_subblocks, dropout_prob) for hidden_size in hidden_sizes],
        nn.Linear(hidden_sizes[-1], 1),
    )
    optimizer = partial(torch.optim.Adam, lr=learning_rate)
    logger.debug(f"Using learning rate {learning_rate} for training")
    # Note to self: good test for finetuning! Would not work with even lr/100
    finetuning_optimizer = partial(torch.optim.Adam, lr=learning_rate/finetuning_factor)
    logger.debug(f"Using learning rate {finetuning_factor} times smaller for finetuning")
    
    return TorchTrainer(
        model, 
        optimizer, 
        output_dir, 
        num_epochs=num_epochs,
        finetune_opt_gen=finetuning_optimizer,
    )

def get_catboost_model(
        feature_dim: int,
        output_dir: Path,
        num_epochs: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.03,
    ):
    return CatBoostTrainer(
        output_dir=output_dir,
        num_epochs=num_epochs,
        depth=depth,
        learning_rate=learning_rate,
    )

def get_xgboost_model(
        feature_dim: int,
        output_dir: Path,
        num_epochs: int = 1000,
        early_stopping: int = 10,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        reg_lambda: Optional[float] = 1,
        reg_alpha: Optional[float] = 0,
        reg_gamma: Optional[float] = 0,
    ):
    return XGBoostTrainer(
        output_dir=output_dir,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_gamma=reg_gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )
