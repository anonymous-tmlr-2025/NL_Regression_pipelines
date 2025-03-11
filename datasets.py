"""
All loaders should return three dataframes (train, val, test) 
with two columns: `text` and `label`.
"""

import csv
import numpy as np
import pandas as pd
from constants import (
    ca_housing, jc_penney, online_boats, online_boats_no_foreign_languages, test_dataset, TEXT_COLUMN, LABEL_COLUMN
)
import gensim.downloader as api
from sklearn.model_selection import train_test_split

def read_csv(path: str):
    return pd.read_csv(path, na_filter=False, quoting=csv.QUOTE_NONNUMERIC)

def load_test_dataset(seed: int):
    """
    Load the test dataset
    """
    def yield_subset(num_samples: int, truncation_length: int):
        text8 = api.load("text8")
        text8 = iter(text8)
        for i in range(num_samples):
            item = next(text8)
            yield " ".join(item[:truncation_length])
    text_data = pd.Series(yield_subset(100, 500))
    # Set the random seed to 1 to make the response always the same for a given text
    rng = np.random.default_rng(1)
    random_response = pd.Series(rng.normal(size=len(text_data)))
    data = pd.concat([text_data, random_response], axis=1)
    data.columns = [TEXT_COLUMN, LABEL_COLUMN]
    # Deliberately set the random seed to 1 to make the test set always the same
    train_val, test = train_test_split(data, test_size=0.2, random_state=1)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    return train, val, test

def load_jc_penney(seed: int):
    test = read_csv("datasets/jc_penney_products/test.csv")
    train_val = read_csv("datasets/jc_penney_products/train.csv")
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    return train, val, test

def load_online_boats(seed: int, foreign_languages_removed: bool = False):
    if foreign_languages_removed:
        suffix = "_no_foreign_languages"
    else:
        suffix = ""
    test = read_csv(f"datasets/online_boat_listings{suffix}/test.csv")
    train_val = read_csv(f"datasets/online_boat_listings{suffix}/train.csv")
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)
    return train, val, test

def load_dataset(dataset: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset
    """
    if dataset == jc_penney:
        return load_jc_penney(seed)
    elif dataset == online_boats:
        return load_online_boats(seed)
    elif dataset == online_boats_no_foreign_languages:
        return load_online_boats(seed, foreign_languages_removed=True)
    elif dataset == test_dataset:
        return load_test_dataset(seed)
    else:
        raise NotImplementedError(f"Dataset {dataset} has no registered loader")