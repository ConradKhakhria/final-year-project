# mypy: ignore-errors
import datasets
import itertools
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import *
import sys
from typing import Tuple

import config
import model


# copied directly from experiment-1-shallow.py
class DatasetLoader:
    def __init__(self, buckets: np.ndarray, seed: int | None = None):
        config.debug("Loading datasets")
        self.dataset = datasets.load_dataset("blog_authorship_corpus", trust_remote_code=True)

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


class Experiment1:
    def __init__(self, buckets: np.ndarray, num_samples: int | None = None):
        config.debug("Loading training and validation sets")

        self.dataset = DatasetLoader(buckets, seed=42)
        self.X_test, self.y_test = self._format_dataset(num_samples)


    def run_experiment(self, model_name: str, pre_prompt_filename: str) -> Tuple[dict, dict]:
        """
        Runs the experiment on a model and pre prompt

        returns:
        a dictionary for age prediction evaluation and gender prediction evaluation
        """
        # This is horrible
        config.SMALL_MODEL = model_name
        isolator = model.BatchModelIsolator("small")

        y_pred_strings = isolator.process_prompts(
            prompts=self.X_test,
            batch_size=20,
            cfg={
                "max_new_tokens": 20,
                "pre_prompt_name": pre_prompt_filename,
                "enforce_json": True
            }
        )
        y_pred = model.extract_json(y_pred_strings, {"age": None, "gender": None})
        y_pred_df = pd.DataFrame(y_pred)
        y_true_df = pd.DataFrame(list(self.y_test))

        print(self.y_test)
        print(y_true_df)
        print(y_pred_df)

        age_evaluation = self._create_evaluation(y_pred_df, y_true_df, "age",
                                                 model_name, pre_prompt_filename)
        gender_evaluation = self._create_evaluation(y_pred_df, y_true_df, "gender",
                                                    model_name, pre_prompt_filename)

        return age_evaluation, gender_evaluation


    def _format_dataset(self, num_samples: int | None = None) -> tuple:
        """
        Formats the dataset for use by the LLM

        This does two things:
        1. Combines age and gender into a single array of dicts
        2. Sorts them by string length
        """
        X_test, y_test_age = self.dataset.get_Xy("test", "age", subset_size=num_samples)
        _, y_test_gender   = self.dataset.get_Xy("test", "gender", subset_size=num_samples)

        # combine age and gender
        y_test = [{"age": a, "gender": g} for a, g in zip(y_test_age, y_test_gender)]

        # sort by string length
        idxs = X_test.str.len().argsort()
        X_test = X_test.iloc[idxs].reset_index(drop=True)
        y_test = [y_test[i] for i in idxs]

        return X_test, np.array(y_test)


    def _create_evaluation(
        self, y_pred_df: pd.DataFrame, y_true_df: pd.DataFrame,
        label: str, model_name: str, pp_filename: str
    ) -> dict:
        """
        Creates an evaluation dictionary for the results and a given label
        """
        valid_idxs = y_pred_df[label].notnull()
        y_pred = y_pred_df[label][valid_idxs]
        y_true = y_true_df[label][valid_idxs]

        return {
            "model": model_name,
            "pre_prompt": pp_filename,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion": confusion_matrix(y_true, y_pred),
            "valid_json": valid_idxs.astype(int).sum() / len(y_pred_df)
        }


if __name__ == "__main__":
    mp.set_start_method("spawn")

    buckets = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
    expt1 = Experiment1(buckets, num_samples=50)

    # Parameters to test
    prompt_names = [
        "expt1-zero-shot.txt",
#        "expt1-few-shot.txt"
    ]

    model_names = [
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]

    # Evaluate
    age_results = []
    gender_results = []

    for model_name, preprompt_filename in itertools.product(model_names, prompt_names):
        age, gender = expt1.run_experiment(model_name, preprompt_filename)
        age_results.append(age)
        gender_results.append(gender)

    df_age = pd.DataFrame(age_results)
    df_gender = pd.DataFrame(gender_results)

    df_age.to_csv(config.RESULTS_DIR / f"age_evaluation_llm.csv")
    df_gender.to_csv(config.RESULTS_DIR / f"gender_evaluation_llm.csv")
