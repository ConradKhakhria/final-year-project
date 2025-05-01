import json
from pathlib import Path
from typing import Dict, List, Tuple

from groq import Groq
import numpy as np
import pandas as pd
from sklearn.metrics import *

import config
import dataset


class Experiment1:
    def __init__(self, *, age_ranges: List[str], genders: List[str], pre_prompt: str, seed: int):
        self.age_ranges = age_ranges
        self.genders = genders
        self.age_range_midpoints = self.get_age_range_midpoints(age_ranges)

        self.dataset = dataset.DatasetLoader("blog_authorship_corpus", np.array(age_ranges), seed=seed)
        self.X, self.y = self.get_Xy(self.dataset)

        # Directories
        self.base_dir = Path.home() / "UCL" / "FYP" / "code"
        self.output_directory = self.base_dir / "groq-results" / "experiment-1"
        self.output_directory.mkdir(parents=True, exist_ok=True)

        api_key = (self.base_dir / "groq-access-token.txt").read_text().strip()
        self.client = Groq(api_key=api_key)

        self.pre_prompt = (self.base_dir / "pre-prompts" / pre_prompt).read_text()


    def set_pre_prompt(self, pp_name: str):
        self.pre_prompt = (self.base_dir / "pre-prompts" / pp_name).read_text()


    # ===== Preprocessing ===== #

    def get_age_range_midpoints(self, age_ranges: List[str]) -> Dict[str, float]:
        """
        Returns a dict mapping each age range to its midpoint
        """
        midpoints = {}

        for r in age_ranges:
            midpoints[r] = sum(map(int, r.split("-"))) / 2

        return midpoints


    def get_Xy(self, dl: dataset.DatasetLoader) -> tuple:
        """
        Returns the correct X and y vectors
        """
        X, y_age = dl.get_Xy("test", "age")
        _, y_gender = dl.get_Xy("test", "gender")

        y = np.array([{ "age": y_age[i], "gender": y_gender[i]} for i in range(len(y_age)) ])

        return X, y


    # ===== Experiment ===== #

    def batch_to_prompt(self, batch: np.ndarray) -> List[dict]:
        """
        Turns a list of posts into a single prompt
        """
        user_message = ""

        for i, post in enumerate(batch):
            user_message += f"[input {i + 1}]: {post}\n"

        return [
            { "role": "system", "content": self.pre_prompt },
            { "role": "user", "content": user_message}
        ]


    def create_evaluation(
        self, y_pred_df: pd.DataFrame, y_true_df: pd.DataFrame, label: str
    ) -> dict:
        """
        Creates a numeric evaluation
        """
        valid_idxs = y_pred_df[label].notnull()

        if label == "age":
            valid_format_idxs = y_pred_df[label].isin(self.age_ranges)
        else:
            valid_format_idxs = y_pred_df[label].isin(self.genders)

        y_pred = y_pred_df[label][valid_format_idxs]
        y_true = y_true_df[label][valid_format_idxs]

        valid_json = valid_idxs.astype(int).sum() / len(y_pred_df)
        valid_format = valid_format_idxs.astype(int).sum() / len(y_pred_df)

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion": confusion_matrix(y_true, y_pred),
            "valid_json": f"{100 * valid_json}%",
            "valid_format": f"{100 * valid_format}%"
        }

        if label == "age":
            y_true_mid = np.array([self.age_range_midpoints[y] for y in y_true])
            y_pred_mid = np.array([self.age_range_midpoints[y] for y in y_pred])
            results["mae"] = np.mean(np.abs(y_true_mid - y_pred_mid))

            y_test_indices = np.array([list(self.dataset.buckets).index(y) for y in y_true])
            y_pred_indices = np.array([list(self.dataset.buckets).index(y) for y in y_pred])
            adjacent_correct = np.sum(np.abs(y_test_indices - y_pred_indices) <= 1)
            results["adjacent_accuracy"] = adjacent_correct / len(y_true)

        return results


    def run_experiment(self, *, model_name: str, batch_size: int) -> Tuple[dict, dict]:
        """
        Runs the experiment with a given batch size

        returns:
        For each of age and gender, a dictionary with an evaluation
        """
        # 1. Make predictions
        predictions = []

        for batch_start in np.arange(0, len(self.X), batch_size):
            print(f"computing batch {batch_start // batch_size} of {len(self.X) // batch_size}")
            batch = self.X[batch_start : batch_start + batch_size]
            messages = self.batch_to_prompt(batch)
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=500
            )

            # Parse output
            try:
                json_string = completion.choices[0].message.content
                predictions.extend(json.loads(json_string))
            except:
                config.debug("Failed to decode json")
                predictions.extend([{ "age": None, "gender": None } for _ in range(batch_size)])


        # 2. Evaluate predictions
        y_pred_df = pd.DataFrame(predictions)
        y_true_df = pd.DataFrame(self.y)

        age_evaluation = self.create_evaluation(y_pred_df, y_true_df, "age")
        gender_evaluation = self.create_evaluation(y_pred_df, y_true_df, "gender")

        return age_evaluation, gender_evaluation



if __name__ == "__main__":
    age_ranges = [f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))]
    genders = ["female", "male"]

    pre_prompt = "expt1-few-shot-groq.txt"

    experiment = Experiment1(
        age_ranges=age_ranges,
        genders=genders,
        pre_prompt=pre_prompt,
        seed=42
    )

    age_evaluation, gender_evaluation = experiment.run_experiment(
        model_name="llama-3.3-70b-versatile",
        batch_size=10
    )

    with open(experiment.output_directory / "age-evaluation.json", "w") as f:
        json.dump(age_evaluation, f)

    with open(experiment.output_directory / "gender-evaluation.json", "w") as f:
        json.dump(gender_evaluation, f)



