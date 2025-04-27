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
import dataset
import model


# These globals govern which phase of the experiment
# is being executed
NUM_SAMPLES = None
OPTIMAL_CONFIGURATION = {
    "model": "mistral",
    "pre-prompt": "expt1-zero-shot.txt"
}


class Experiment1:
    def __init__(self, buckets: np.ndarray, num_samples: int | None = None):
        config.debug("Loading training and validation sets")

        self.dataset = dataset.DatasetLoader("blog_authorship_corpus", buckets, seed=42)
        self.X_test, self.y_test = self._format_dataset(num_samples)

        self.bucket_midpoints = {}
        for b in buckets:
            self.bucket_midpoints[b] = sum(map(int, b.split("-"))) / 2

        # Output directories
        self.string_output = config.RESULTS_DIR / "string-output"
        self.errors = config.RESULTS_DIR / "errors"
        self.evaluation = config.RESULTS_DIR / "evaluation"

        self.string_output.mkdir(parents=True, exist_ok=True)
        self.errors.mkdir(parents=True, exist_ok=True)
        self.evaluation.mkdir(parents=True, exist_ok=True)


    def run_experiment(
        self, model_name: str, pre_prompt_filename: str, batch_size: int
    ) -> Tuple[dict, dict]:
        """
        Runs the experiment on a model and pre prompt

        args:
        - model_name: the name of the model to use
        - pre_prompt_filename: the pre-prompt to use
        - batch_size: the number of prompts to run simultaneously

        returns:
        a dictionary for age prediction evaluation and gender prediction evaluation
        """
        isolator = model.BatchModelIsolator(model_name)

        y_pred_strings = isolator.process_prompts(
            prompts=self.X_test,
            batch_size=batch_size,
            cfg={
                "max_new_tokens": 30,
                "pre_prompt_name": pre_prompt_filename,
                "enforce_json": True
            }
        )

        # Record the actual string outputs:
        model_short_name = model_name.split("/")[1]
        pre_prompt_short_name = pre_prompt_filename[:-4]
        out_file = self.string_output / f"{model_short_name}-{pre_prompt_short_name}.txt"

        with open(out_file, "w") as f:
            for i, s in enumerate(y_pred_strings):
                f.write(f"string {i + 1}:\n{s}\n\n")

        y_pred = model.extract_json(y_pred_strings, {"age": None, "gender": None})
        y_pred_df = pd.DataFrame(y_pred)
        y_true_df = pd.DataFrame(list(self.y_test))

        age_evaluation = self._create_evaluation(y_pred_df, y_true_df, "age",
                                                 model_name, pre_prompt_filename)
        gender_evaluation = self._create_evaluation(y_pred_df, y_true_df, "gender",
                                                    model_name, pre_prompt_filename)

        del isolator

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
        if not {"age", "gender"}.issubset(y_pred_df.columns):
            model_short_name = model_name.split("/")[1]
            pre_prompt_short_name = pp_filename[:-4]
            y_pred_df.to_csv(self.errors / f"{model_short_name}_{pre_prompt_short_name}.csv")

            return {
                "model": model_name,
                "pre_prompt": pp_filename[:-4],
                "accuracy": None,
                "f1_macro": None,
                "confusion": None,
                "valid_json": None,
                "valid_format": None
            }

        valid_idxs = y_pred_df[label].notnull()

        if label == "age":
            valid_format_idxs = y_pred_df[label].isin(self.dataset.buckets)
        else:
            valid_format_idxs = y_pred_df[label].isin(["male", "female"])

        y_pred = y_pred_df[label][valid_format_idxs]
        y_true = y_true_df[label][valid_format_idxs]

        valid_json = valid_idxs.astype(int).sum() / len(y_pred_df)
        valid_format = valid_format_idxs.astype(int).sum() / len(y_pred_df)

        results = {
            "model": model_name,
            "pre_prompt": pp_filename[:-4],
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion": confusion_matrix(y_true, y_pred),
            "valid_json": f"{100 * valid_json}%",
            "valid_format": f"{100 * valid_format}%"
        }

        if label == "age":
            y_true_mid = np.array([self.bucket_midpoints[y] for y in y_true])
            y_pred_mid = np.array([self.bucket_midpoints[y] for y in y_pred])
            results["mae"] = np.mean(np.abs(y_true_mid - y_pred_mid))

            y_test_indices = np.array([list(self.dataset.buckets).index(y) for y in y_true])
            y_pred_indices = np.array([list(self.dataset.buckets).index(y) for y in y_pred])
            adjacent_correct = np.sum(np.abs(y_test_indices - y_pred_indices) <= 1)
            results["adjacent_accuracy"] = adjacent_correct / len(y_true)

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn")

    buckets = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
    expt1 = Experiment1(buckets, num_samples=NUM_SAMPLES)

    # Parameters to test
    prompt_names = [
        "expt1-zero-shot.txt",
        "expt1-few-shot.txt"
    ]

    model_names = [
        "mistral",
        "llama-2",
        "deepseek",
    ]

    model_configs = {
        "mistral": {
            "full_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "batch_size": 200
        },
        "llama-2": {
            "full_name": "meta-llama/Llama-2-7b-chat-hf",
            "batch_size": 50
        },
        "deepseek": {
            "full_name": "deepseek-ai/deepseek-llm-7b-chat",
            "batch_size": 100
        }
    }

    if OPTIMAL_CONFIGURATION is not None:
        model_names = [OPTIMAL_CONFIGURATION["model"]]
        prompt_names = [OPTIMAL_CONFIGURATION["pre-prompt"]]

    # Evaluate
    age_results = []
    gender_results = []

    for model_name, preprompt_filename in itertools.product(model_names, prompt_names):
        full_name = model_configs[model_name]["full_name"]
        batch_size = model_configs[model_name]["batch_size"]
        age, gender = expt1.run_experiment(full_name, preprompt_filename, batch_size)
        age_results.append(age)
        gender_results.append(gender)

    df_age = pd.DataFrame(age_results)
    df_gender = pd.DataFrame(gender_results)

    # Filenames
    if NUM_SAMPLES is not None:
        df_age_name = f"age_evaluation_llm_{NUM_SAMPLES}_samples.csv"
        df_gender_name = f"gender_evaluation_llm_{NUM_SAMPLES}_samples.csv"
    else:
        df_age_name = "age_evaluation_llm.csv"
        df_gender_name = "gender_evaluation_llm.csv"

    df_age.to_csv(expt1.evaluation / df_age_name)
    df_gender.to_csv(expt1.evaluation / df_gender_name)

    # Print evaluation metrics to console
    print("\n===== AGE PREDICTION RESULTS =====")
    print(df_age)
    print("\n===== GENDER PREDICTION RESULTS =====")
    print(df_gender)
