# mypy: ignore-errors
import datasets
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Tuple

import config


class DatasetLoader:
    def __init__(self, dataset_name: str, buckets: np.ndarray, seed: int | None = None):
        config.debug("Loading datasets")
        self.dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)

        self.df_train = self.dataset["train"].to_pandas()
        self.df_test = self.dataset["validation"].to_pandas()

        self.X = {
            "train": self.df_train["text"],
            "test": self.df_test["text"]
        }

        self.y = {
            "train": { "age": self.df_train["age"], "gender": self.df_train["gender"] },
            "test":  { "age": self.df_test["age"],  "gender": self.df_test["gender"]}
        }

        self.buckets = buckets
        self.rng = np.random.default_rng(seed)


    def get_Xy(
        self, split: str, label: str, subset_size: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the X and y pair

        args:
        - split: 'train' or 'test'
        - label: 'age' or 'gender'
        - subset_size: the size of the random subset to use (default None: full dataset)

        returns:
        the tuple containing:
            - X
            - y
        """
        if split not in ["train", "test"]:
            raise NameError(f"No split named '{split}'")

        if label not in ["age", "gender"]:
            raise NameError(f"No target label '{label}'")

        X = self.X[split]
        y = self.y[split][label]

        if label == "age":
            y = self.buckets[y]

        if subset_size is not None:
            idxs = self.rng.choice(np.arange(len(X)), size=subset_size, replace=False)

            X = X[idxs]
            y = y[idxs]

        return X, y


@config.debug_function
def load_dataset_locally(dataset_name: str, sort_axis=None, split_set="train") -> pd.DataFrame:
    """
    Loads a dataset from huggingface

    If the dataset has been cached, it is loaded locally.
    Otherwise, this function caches it locally.

    Args:
    - dataset_name: name of the dataset on Hugging Face hub
    - X_name: name of the input (X) column
    - y_names: list of output (y) column names
    - sort_axis: column to sort by length (usually a text field)
    - split_set: which of "train", "validation", and "test" you want

    Returns:
        the pandas DataFrame of the dataset
    """

    if split_set not in {"train", "validation"}:
        raise SyntaxError(f"split_set must be one of 'train' or 'validation'")

    dataset_cache_path = config.CACHE_DIR / "huggingface-datasets"
    local_dataset_path = config.DATASET_DIR / f"{dataset_name}_local_{split_set}.parquet"

    local_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if local_dataset_path.exists():
        config.debug("Loading cached dataset")
        df = pd.read_parquet(local_dataset_path)
    else:
        config.debug("Loading dataset from huggingface")
        dataset = datasets.load_dataset(dataset_name,
                                        split=split_set,
                                        cache_dir=str(dataset_cache_path),
                                        trust_remote_code=True)
        df = dataset.to_pandas()

        config.debug("Sorting dataset")
        if sort_axis is not None:
            df[f"{sort_axis}_length"] = df[sort_axis].str.len()
            df = df.sort_values(f"{sort_axis}_length").reset_index(drop=True)

        config.debug(f"Caching dataset locally to {local_dataset_path}")
        df.to_parquet(local_dataset_path)

    return df
