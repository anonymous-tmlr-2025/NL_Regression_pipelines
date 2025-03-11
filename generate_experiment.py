import hashlib
from itertools import chain, combinations, product
import json
from pathlib import Path
from filelock import FileLock
import random

from loguru import logger
from constants import FEATURISERS_CAPABLE_OF_FINETUNING, MODELS_CAPABLE_OF_FINETUNING, SEED
from pipeline_components import (
    iterate_component_names,
    preprocessers,
    tokenisers,
    featurisers,
    models,
)

def validate_config(finetune, tokeniser, featuriser, model, preprocesser_order):
    if (
        ("stem" in preprocesser_order) and
        ("lemmatize" in preprocesser_order)
    ):
        raise ValueError("Stemmer and lemmatizer cannot be used together")
    if featuriser in ["bert_cls", "bert_mean"]:
        if tokeniser != "bert":
            raise ValueError("BERT featurisers require BERT tokeniser")

    if featuriser == "glove_mean":
        if tokeniser not in [
            "whitespace", "stanford", 
            # "treebank", # Unclear if Treebank is compatible with GloVe
        ]:
            raise ValueError("GloVe mean featuriser requires whitespace, stanford or treebank tokeniser")
    
    if finetune:
        if (
            featuriser not in FEATURISERS_CAPABLE_OF_FINETUNING or
            model not in MODELS_CAPABLE_OF_FINETUNING
        ):
            raise ValueError(
                "Only deep learning models can be finetuned\n"
                "Please choose `bert_cls` or `bert_mean` as featuriser"
                " and `mlp` or `resnet` as model."
            )


def generate_random_experiment_config(
        rng_seed: int = SEED,
        exp_seed: int = SEED,
        include_preprocessers: bool = True,
    ):
    """
    Generate a random valid experiment config.
    """
    # Don't allow completely random seeds
    if rng_seed is None:
        logger.warning(f"Not allowed intentionally no seed provided - using default seed {SEED}")
        rng_seed = SEED
    if not include_preprocessers:
        logger.warning("Are you sure you want to generate random config without any preprocessers? At that point may as well do deterministic")
    # Work with just names as we will give this back to the pipeline to re-fetch
    # essentially we are acting like a cli user.
    hard_limit = 10000
    preprocesser_names = list(iterate_component_names(preprocessers))
    tokeniser_names = list(iterate_component_names(tokenisers))
    featuriser_names = list(iterate_component_names(featurisers))
    model_names = list(iterate_component_names(models))
    rng = random.Random()
    rng.seed(a=rng_seed)
    suggested = set()
    attempts = 0
    while True:
        attempts += 1
        if attempts > hard_limit:
            raise ValueError(f"Preventing infinite loop - hard limit of {hard_limit} reached")
        rng.shuffle(preprocesser_names)
        if not include_preprocessers:
            chosen_preprocessers = [] 
        else:
            chosen_preprocessers = preprocesser_names[:rng.randint(0, len(preprocesser_names))]
        featuriser = rng.choice(featuriser_names)
        model = rng.choice(model_names)

        # Handle the case where we have a featuriser that requires a specific tokeniser
        if featuriser in ["bert_cls", "bert_mean"]:
            tokeniser = "bert"
        elif featuriser in ["glove_mean"]:
            tokeniser = rng.choice(["whitespace", "stanford"])
        else:
            tokeniser = rng.choice(tokeniser_names)
        
        
        # Handle the case where we have multiple preprocessers that do the same thing
        if "stem" in chosen_preprocessers and "lemmatize" in chosen_preprocessers:
            chosen_preprocessers.remove(rng.choice(["stem", "lemmatize"]))
        if "no_numbers_simple" in chosen_preprocessers and "no_numbers_spacy" in chosen_preprocessers:
            chosen_preprocessers.remove(rng.choice(["no_numbers_simple", "no_numbers_spacy"]))
        if "no_stopwords_spacy" in chosen_preprocessers and "no_stopwords_nltk" in chosen_preprocessers:
            chosen_preprocessers.remove(rng.choice(["no_stopwords_spacy", "no_stopwords_nltk"]))
        
        if (model in MODELS_CAPABLE_OF_FINETUNING) and (featuriser in FEATURISERS_CAPABLE_OF_FINETUNING):
            finetune = rng.choice([True, False])
        else:
            finetune = False
        
        config = {
            "preprocesser_order": recursive_sort(chosen_preprocessers),
            "tokeniser": tokeniser,
            "featuriser": featuriser,
            "model": model,
            "finetune": finetune,
            "seed": exp_seed,
        }
        hash = hash_experiment_config(config)
        if hash in suggested:
            continue
        # To add new keys in the future we can do so without breaking backwards compatibility
        # by just adding them to the config if they exist.
        yield config
        # Keep track of what we have suggested to avoid suggesting the same thing twice
        suggested.add(hash)

def power_set(iterable):
    """
    Return all subsets of the given iterable.
    """
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1))

def resample_with_different_seed(
    new_seed: int, 
    seed_to_replicate: int,
    log_path: Path,
    include_preprocessers: bool = True, 
    verbose=False
    ):
    """
    Resample with a different seed, skipping experiments that use disabled components.
    """
    if new_seed == seed_to_replicate:
        raise ValueError("Cannot resample with the same seed")
    if not include_preprocessers:
        raise NotImplementedError("Cannot yet resample without preprocessers")

    existing = json.loads(log_path.read_text())
    observed_hashes = set()
    warned_about_multiple_seeds = False

    # Get currently available component names
    available_preprocessers = set(iterate_component_names(preprocessers))
    available_tokenisers = set(iterate_component_names(tokenisers))
    available_featurisers = set(iterate_component_names(featurisers))
    available_models = set(iterate_component_names(models))

    for _, config in existing.items():
        del config["status"]
        del config["error"]
        old_seed = config["seed"]
        # Skip if the seed is not the one we are replicating
        if old_seed != seed_to_replicate:
            continue

        # Skip if config uses any disabled components
        if verbose:
            warning_logger = logger.warning
        else:
            warning_logger = logger.debug
        if any(p not in available_preprocessers for p in config["preprocesser_order"]):
            warning_logger(f"Skipping config using disabled preprocesser: {config['preprocesser_order']}")
            continue
        if config["tokeniser"] not in available_tokenisers:
            warning_logger(f"Skipping config using disabled tokeniser: {config['tokeniser']}")
            continue
        if config["featuriser"] not in available_featurisers:
            warning_logger(f"Skipping config using disabled featuriser: {config['featuriser']}")
            continue
        if config["model"] not in available_models:
            warning_logger(f"Skipping config using disabled model: {config['model']}")
            continue

        del config["seed"]
        no_seed_hash = hash_experiment_config(config)
        if no_seed_hash in observed_hashes:
            continue
        observed_hashes.add(no_seed_hash)
        config["seed"] = new_seed
        yield config

def generate_experiment_config(exp_seed, include_preprocessers: bool = True, verbose=False):
    """
    Here seed does nothing as we have deterministic experiment selection.
    """ 
    _tokenisers = list(iterate_component_names(tokenisers))
    _featurisers = list(iterate_component_names(featurisers))
    _models = list(iterate_component_names(models))
    _preprocessers = list(iterate_component_names(preprocessers))
    finetune = [False, True]
    choices = [finetune, _models, _featurisers, _tokenisers]
    if include_preprocessers:
        choices.append(power_set(_preprocessers))
    else:
        choices.append([[]])
    for finetune, model, featuriser, tokeniser, preprocesser_order in product(*choices):
        try:
            validate_config(finetune, tokeniser, featuriser, model, preprocesser_order)
        except ValueError as e:
            if verbose:
                logger.debug(f"Invalid config: {finetune}, {model}, {featuriser}, {tokeniser}, {preprocesser_order} - {e}")
            continue
        config = {
            "preprocesser_order": preprocesser_order,
            "tokeniser": tokeniser,
            "featuriser": featuriser,
            "model": model,
            "seed": exp_seed,
        }
        if finetune:
            config["finetune"] = finetune
        yield config

def get_experiment_that_has_not_been_run(
        experiment_log: Path,
        random_selection:bool = False,
        bootstrap_resampling: bool = False,
        max_attempts: int = 100000,
        exp_seed: int = SEED,
        seed_to_replicate: int = SEED,
        random_order_seed: int = SEED,
        skip: int = 0,
        include_preprocessers: bool = True,
        verbose: bool = False,
):
    """
    Get an experiment that has not been run, checks experiment log for completed
    experiments and if generating random experiments will try up to max_attempts
    times before giving up.

    `random_order_seed` is only used if `random_selection` is true and controls 
    the order of the random experiments by seeding the generator.
    """
    if random_selection and bootstrap_resampling:
        raise ValueError("Cannot have both random selection and bootstrap resampling")
    
    if random_selection:
        experiment_generator = generate_random_experiment_config(random_order_seed, exp_seed, include_preprocessers)
    elif bootstrap_resampling:
        experiment_generator = resample_with_different_seed(exp_seed, seed_to_replicate, experiment_log, include_preprocessers)
    else:
        experiment_generator = generate_experiment_config(exp_seed, include_preprocessers)
    
    logger.info(f"Getting experiment that has not been run with skip {skip}, at most {max_attempts} attempts")
    attempts = 0
    looked_through = 0
    # While not picked an experiment
    while True:
        # Generate an experiment config
        try:
            experiment_config = next(experiment_generator)
        except StopIteration:
            raise ValueError("No valid experiments found that have not been run - check experiment log")
        if verbose:
            logger.debug(f"Generated experiment config: {experiment_config}")
        # Check for its existence in the experiment log
        if experiment_is_done(experiment_log, experiment_config):
            attempts += 1
            if attempts > max_attempts:
                raise ValueError("Max attempts reached - Could be ok if this was set as a small limit")
        else:
            if skip > 0:
                skip -= 1
                continue
            logger.info(f"Found experiment that has not been run after {attempts} attempts and looked through {looked_through} experiments")
            return experiment_config
        looked_through += 1

        
def recursive_sort(node) -> dict:
    if isinstance(node, dict):
        # Sort keys and have children so sort them
        return {k: recursive_sort(v) for k, v in sorted(node.items())}
    else:
        # If iterable, sort it
        if hasattr(node, '__iter__') and not isinstance(node, (str, bytes)):
            return sorted(node)
        # If not iterable, return as is
        return node


def hash_experiment_config(experiment_config: dict) -> str:
    # recursively sort experiment config for consistent hashing
    experiment_config = recursive_sort(experiment_config)
    config_str = json.dumps(experiment_config).encode()
    return hashlib.sha256(config_str, usedforsecurity=False).hexdigest()

def experiment_is_done(experiment_log: Path, experiment_config: dict) -> bool:
    """
    Check if an experiment has been completed by checking the experiment log.

    Essentially a convenience function for correctly checking hashes.
    """
    experiment_config_hash = hash_experiment_config(experiment_config)
    # Really generous timeout of 10 minutes
    lock = FileLock(experiment_log.with_suffix(".lock"), timeout=60*10)
    with lock:
        prior_experiment_hashes = json.loads(experiment_log.read_text())
    return experiment_config_hash in prior_experiment_hashes

def update_experiment_log(experiment_log: Path, experiment_config: dict, error: str | None = None):
    """
    Update the experiment log with the status of an experiment.

    Essentially a convenience function for correctly updating the experiment
    log and hashing the experiment config.
    """
    # Really generous timeout of 10 minutes
    logger.info(f"Updating experiment log {experiment_log}")
    lock = FileLock(experiment_log.with_suffix(".lock"), timeout=60*10)
    experiment_config_hash = hash_experiment_config(experiment_config)
    to_save = experiment_config.copy()
    to_save["preprocesser_order"] = recursive_sort(to_save["preprocesser_order"])
    if error is not None:
        to_save["status"] = "failed"
        to_save["error"] = error
    else:
        to_save["status"] = "success"
        to_save["error"] = None
    with lock:
        prior_experiment_hashes = json.loads(experiment_log.read_text())
        prior_experiment_hashes[experiment_config_hash] = to_save
        experiment_log.write_text(json.dumps(prior_experiment_hashes, indent=0))
    logger.info(f"Updated experiment log {experiment_log}")
