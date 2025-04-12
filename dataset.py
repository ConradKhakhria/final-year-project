# mypy: ignore-errors
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


def load_reddit_submissions_comments(submissions_fname: Path, comments_fname: Path) -> pd.DataFrame:
    """
    Loads a df containing all submissions and comments
    """
    print(f"submissions_fname = {submissions_fname}")

    submissions = pd.read_json(submissions_fname, lines=True)
    comments = pd.read_json(comments_fname, lines=True)

    # create 'text' columns for both
    submissions["text"] = submissions["title"].fillna("") + "\n" + submissions["selftext"].fillna("")
    comments["text"] = comments["body"]

    comments["type"] = "comment"
    submissions["type"] = "submission"

    selected_columns = ["id", "author", "created_utc", "subreddit", "score", "text", "type"]

    comments_reduced = comments[selected_columns]
    submissions_reduced = submissions[selected_columns]
    
    # Combine the two DataFrames.
    combined_df = pd.concat([comments_reduced, submissions_reduced], ignore_index=True)
    
    return combined_df
