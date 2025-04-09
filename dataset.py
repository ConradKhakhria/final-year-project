import datasets
import os
import pandas as pd
from pathlib import Path

import config


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

    dataset_cache_path = config.CACHE_DIR / "huggingface-datasets"
    local_dataset_path = config.DATASET_DIR / f"{dataset_name}_local.parquet"

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
