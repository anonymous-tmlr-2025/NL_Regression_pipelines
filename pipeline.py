import json
from pathlib import Path
import time
from typing import Annotated, Optional
from loguru import logger
import numpy as np
import pandas as pd
import yaml
import typer

from datasets import load_dataset
from metrics import METRICS_TO_RECORD
from constants import (
    FEATURISERS_CAPABLE_OF_FINETUNING,
    MANDATORY_CONFIG_KEYS,
    FINETUNE_OPT,
    LABEL_COLUMN,
    MODELS_CAPABLE_OF_FINETUNING,
    OUTPUT_DIR,
    SEED,
    TEXT_COLUMN,
    TOKENISER_CONFIG,
    FEATURISER_CONFIG,
    MODEL_CONFIG,
    PREPROCESSER_ORDER,
)

from pipeline_components import (
    PipelineComponent,
    filter_components_by_name,
    preprocessers,
    tokenisers,
    featurisers,
    models,
)

from generate_experiment import (
    get_experiment_that_has_not_been_run,
    update_experiment_log,
    validate_config,
)
from pipeline_models import TorchTrainer

def get_X_y(df: pd.DataFrame):
    # Preprocessers and tokenisers expect a pandas series
    X = df[TEXT_COLUMN]
    # Lets guarantee that label is 2D to avoid any confusion
    y = df[[LABEL_COLUMN]].values
    return X, y

def path_from_config(
        root_dir: Path,
        chosen_preprocessers: list[str],
        tokeniser: str,
        featuriser: str,
        model: str,
        seed: int,
        finetune: bool,
):
    output_dir = root_dir
    if chosen_preprocessers:
        output_dir = output_dir / f"preprocessers_{''.join(chosen_preprocessers)}"
    output_dir = output_dir / f"tokeniser_{tokeniser}"
    output_dir = output_dir / f"featuriser_{featuriser}"
    output_dir = output_dir / f"model_{model}"
    output_dir = output_dir / f"seed_{seed}"
    if finetune:
        output_dir = output_dir / "finetune"
    return output_dir

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def cli(
    dataset: Annotated[str, typer.Option(help="Dataset to use")],
    downsample_frac: Annotated[float, typer.Option(help="Fraction of dataset to use")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = SEED, 
    normalise_response: Annotated[bool, typer.Option(help="If true, will normalise the response variable with x-mu/sigma")] = False,
    random_experiment: Annotated[bool, typer.Option(help="If true, will generate a random experiment config")] = False,
    bootstrap: Annotated[bool, typer.Option(help="If true, will repeat experiment log experiments with (new) seed provided")] = False,
    ignore_preprocessers: Annotated[bool, typer.Option(help="If true, will ignore preprocessers in experiment config")] = False,
    skip: Annotated[int, typer.Option(help="Number of experiments to skip when generating a random experiment")] = 0,
    up_to: Annotated[Optional[int], typer.Option(help="Number of completed experiments to stop at")] = None,
    experiment_config: Annotated[Optional[Path], typer.Option(help="Path to config file - if not provided, will select any experiment that has not been run")] = None,
    experiment_log: Annotated[Optional[Path], typer.Option(help="Path to experiment log file - used if no config file provided")] = None,
    output_root_dir: Annotated[Optional[Path], typer.Option(help="Root directory for output files - if not provided, will use 'results'")] = None,
    low_disk: Annotated[bool, typer.Option(help="If true, will not save model to disk")] = False,
):
    return runner(
        dataset=dataset, 
        downsample_frac=downsample_frac, 
        seed=seed, 
        random_experiment=random_experiment, 
        bootstrap=bootstrap, 
        ignore_preprocessers=ignore_preprocessers, 
        skip_experiments=skip, 
        up_to_experiment=up_to, 
        experiment_config=experiment_config, 
        experiment_log=experiment_log, 
        output_root_dir=output_root_dir, 
        low_disk_mode=low_disk,
        normalise_response=normalise_response
    )

def runner(
    dataset: str,
    normalise_response: bool,
    seed: int,
    random_experiment: bool,
    bootstrap: bool,
    ignore_preprocessers: bool,
    skip_experiments: int,
    downsample_frac: Optional[float] = None,
    up_to_experiment: Optional[int] = None,
    experiment_config: Optional[Path] = None,
    experiment_log: Optional[Path] = None,
    output_root_dir: Optional[Path] = None,
    low_disk_mode: bool = False,
):
    """
    Loads in the dataset, creates pipeline from experiment config, creates
    a unique output directory for the experiment and runs the experiment.

    Some basic validation is done on the experiment config to make sure the pipeline can run.

    Attempts to avoid repeating experiments by checking the experiment log;
    which tracks completed experiments via hashing of the experiment config.

    If no config file is provided, will select any experiment that has not been run.
    If specifying random experiment, will generate a random experiment config.
    """
    logger.info(f"Running experiment with seed {seed} on dataset {dataset}")
    if skip_experiments and not random_experiment:
        logger.warning("Skipping experiments but not generating a random experiment - you could skip past all experiments")
    if low_disk_mode:
        logger.warning("Low disk mode is enabled - model will not be saved to disk")
    root = output_root_dir or OUTPUT_DIR
    if downsample_frac is not None:
        logger.info(f"Downsampling dataset to {downsample_frac} of original size")
        root = root / f"downsample_{downsample_frac}"
    if normalise_response:
        logger.info("Normalising response variable")
        root = root / "normalised_response"
    dataset_output_dir = root / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {dataset_output_dir}")
    if experiment_log:
        logger.info(f"Loading experiment log from user specified path: {experiment_log}")
        experiment_log = Path(experiment_log)
        if not experiment_log.exists():
            raise ValueError("No experiment log file found - must provide if not providing config file")
    else:
        experiment_log = dataset_output_dir / f"experiment_log.json"
        logger.info(f"No experiment log provided - using default path: {experiment_log}")
        if not experiment_log.exists():
            experiment_log.write_text(r"{}")

    if experiment_config:
        logger.info(f"Loading config file from {experiment_config}")
        experiment_config = yaml.safe_load(experiment_config.read_text())
    else:
        # Find all experiments that have been run and select any experiment that has not
        logger.info("No config file provided - selecting any experiment that has not been run")
        
        experiment_config = get_experiment_that_has_not_been_run(
            experiment_log=experiment_log, 
            random_selection=random_experiment,
            bootstrap_resampling=bootstrap,
            exp_seed=seed,
            random_order_seed=1,
            skip=skip_experiments,
            max_attempts=up_to_experiment if up_to_experiment else 10000,
            include_preprocessers=not ignore_preprocessers,
        )

    for key in MANDATORY_CONFIG_KEYS:
        assert key in experiment_config, f"Config must contain key {key}"

    logger.info(f"Config: {experiment_config}")
    validate_config(
        experiment_config.get(FINETUNE_OPT, False), 
        experiment_config[TOKENISER_CONFIG], 
        experiment_config[FEATURISER_CONFIG], 
        experiment_config[MODEL_CONFIG], 
        experiment_config[PREPROCESSER_ORDER]
    )

    # Load the dataset
    train, val, test = load_dataset(dataset, seed=seed)
    train_X, train_y = get_X_y(train)
    if downsample_frac is not None:
        rng = np.random.default_rng(1)
        downsample_idx = rng.choice(len(train_X), size=int(len(train_X) * downsample_frac), replace=False)
        train_X, train_y = train_X.iloc[downsample_idx], train_y[downsample_idx]
        logger.info(f"Downsampled training dataset to {len(train_X)} samples")
    val_X, val_y = get_X_y(val)
    test_X, test_y = get_X_y(test)
    if normalise_response:
        mean = train_y.mean()
        std = train_y.std()
        train_y = (train_y - mean) / std
        val_y = (val_y - mean) / std
        test_y = (test_y - mean) / std
        logger.info(f"Normalised response variable with mean {mean} and std {std}")
        normalisation_params = {
            "mean": mean,
            "std": std
        }

    # Take the correct pipeline components for this experiment
    chosen_preprocessers: list[PipelineComponent] = filter_components_by_name(preprocessers, experiment_config[PREPROCESSER_ORDER])
    tokeniser: PipelineComponent = filter_components_by_name(tokenisers, experiment_config[TOKENISER_CONFIG])
    featuriser: PipelineComponent = filter_components_by_name(featurisers, experiment_config[FEATURISER_CONFIG])
    model: PipelineComponent = filter_components_by_name(models, experiment_config[MODEL_CONFIG])

    # Create output directory structure
    output_dir = path_from_config(
        dataset_output_dir, 
        [x.name for x in chosen_preprocessers], tokeniser.name, 
        featuriser.name, model.name, 
        seed, experiment_config.get(FINETUNE_OPT, False)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if normalise_response:
        with open(output_dir / "normalisation_params.json", "w") as f:
            json.dump(normalisation_params, f)
    try:
        ###### THIS IS WHERE THE ACTUAL EXPERIMENT IS RUN ######
        experiment(
            train_X, val_X, test_X, train_y, val_y, test_y, 
            chosen_preprocessers, tokeniser, featuriser, model, 
            output_dir, experiment_config.get(FINETUNE_OPT, False),
            low_disk_mode
        )
    except Exception as e:
        # Register experiment in experiment log
        stringified_error = repr(e)
        logger.error(f"Experiment failed with error: {stringified_error}")
        update_experiment_log(experiment_log, experiment_config, error=stringified_error)
    else:
        update_experiment_log(experiment_log, experiment_config)

def experiment(
    train_X: pd.Series,
    val_X: pd.Series,
    test_X: pd.Series,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    chosen_preprocessers: list[PipelineComponent],
    tokeniser: PipelineComponent,
    featuriser: PipelineComponent,
    model: PipelineComponent,
    output_dir: Path,
    finetune: bool = False,
    low_disk_mode: bool = False,
): 
    """
    Given train, val and test data, run the pipeline comprised of the
    chosen preprocessers, tokeniser, featuriser and model.

    Assumes output_dir is a valid, existing directory that will not conflict
    with any other experiment.

    Saves the model, predictions on trian, val and test data into output_dir.

    Various metrics are evaluated and saved along with experiment config and
    running time in a json file in the output directory.

    If finetune is true, will finetune the model on the training data;
    only available for deep learning models with bert featurisers.
    
    """
    # Run the training
    overall_start_time = time.time()
    preprocessed_train_X = train_X
    preprocessed_val_X = val_X
    preprocessed_test_X = test_X
    for name, fn in chosen_preprocessers:
        preprocesser = fn()
        logger.info(f"Preprocessing with {name}")
        preprocessed_train_X = preprocesser(preprocessed_train_X)
        preprocessed_val_X = preprocesser(preprocessed_val_X)
        preprocessed_test_X = preprocesser(preprocessed_test_X)
        
    logger.info(f"Tokenising with {tokeniser.name}")
    tokenised_train_X = tokeniser.fn()(preprocessed_train_X)
    tokenised_val_X = tokeniser.fn()(preprocessed_val_X)
    tokenised_test_X = tokeniser.fn()(preprocessed_test_X)

    logger.info(f"Featurising with {featuriser.name}")
    trained_featuriser = featuriser.fn(tokenised_train_X)
    featurised_train_X = trained_featuriser(tokenised_train_X)
    featurised_val_X = trained_featuriser(tokenised_val_X)
    featurised_test_X = trained_featuriser(tokenised_test_X)

    # Here is where I need to think about how to handle the model
    # specific args in a way that will be amenable to optuna
    model_start_time = time.time()
    logger.info(f"Training with {model.name}")
    trainer = model.fn(
        feature_dim=featurised_train_X.shape[1],
        output_dir=output_dir,
    )
    trainer.fit(featurised_train_X, train_y, featurised_val_X, val_y, low_disk_mode=low_disk_mode)

    if finetune:
        trainer: TorchTrainer = trainer
        finetune_featuriser = featuriser.fn(tokenised_train_X, for_finetuning=True)
        trainer.finetune(
            finetune_featuriser, tokenised_train_X, train_y, 
            tokenised_val_X, val_y, track_memory=False, low_disk_mode=low_disk_mode
        )

    # Save the model
    if not low_disk_mode:
        trainer.save(output_dir)
    model_end_time = time.time()

    logger.info("Evaluating and saving.")
    # Evaluate on test
    if finetune:
        train_predictions = trainer.predict(tokenised_train_X)
        val_predictions = trainer.predict(tokenised_val_X)
        test_predictions = trainer.predict(tokenised_test_X)   
    else:
        train_predictions = trainer.predict(featurised_train_X)
        val_predictions = trainer.predict(featurised_val_X)
        test_predictions = trainer.predict(featurised_test_X)
    
    def save_predictions(predictions, truth, filename):
        pred_df = pd.DataFrame(
            np.concatenate([predictions, truth], axis=1), 
            columns=["prediction", "label"]
        )
        pred_df.to_csv(output_dir / filename, index=False)
    
    save_predictions(train_predictions, train_y, "train_preds.csv")
    save_predictions(val_predictions, val_y, "val_preds.csv")
    save_predictions(test_predictions, test_y, "test_preds.csv")
    
    results = {
        "preprocessers": [p.name for p in chosen_preprocessers],
        "tokeniser": tokeniser.name,
        "featuriser": featuriser.name,
        "model": model.name,
        "finetune": finetune,
        "total_time": model_end_time - overall_start_time,
        "preprocess_time": model_start_time - overall_start_time,
        "model_time": model_end_time - model_start_time,
    }
    for metric_name, metric_fn in METRICS_TO_RECORD.items():
        train_metric_val = metric_fn(train_y, train_predictions)
        val_metric_val = metric_fn(val_y, val_predictions)
        test_metric_val = metric_fn(test_y, test_predictions)
        logger.info(f"Test {metric_name}: {test_metric_val}")
        results[metric_name] = {
            "train": train_metric_val,
            "val": val_metric_val,
            "test": test_metric_val
        }
    
    # Save the results - make it clear which part of which pipeline was used 
    # to get the results
    with open(output_dir / f"results.json", "w") as f:
        json.dump(results, f)

    
if __name__ == "__main__":
    app()