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


class Experiment1:
    def __init__(self, pre_prompt_filename: str, num_samples: int = None):
        config.debug("Loading training and validation sets")
        self.df_train = dataset.load_dataset_locally("blog_authorship_corpus",
                                                     sort_axis="text", split_set="train")
        self.df_val = dataset.load_dataset_locally("blog_authorship_corpus",
                                                     sort_axis="text", split_set="validate")

        config.debug("Creating testing set")
        np.random.seed(42)

        if num_samples is None:
            self.selected_indices = np.arange(len(self.df_val))
        else:
            self.selected_indices = np.random.choice(np.arange(len(self.df_val)),
                                                     size=num_samples, replace=False)
            self.selected_indices.sort()

        self.df_test = self.df_val.iloc[self.selected_indices]

        self.X_test = self.df_test["text"]
        self.y_test = self.df_test[["age", "gender"]].to_dict(orient="records")

        self.pre_prompt_path = config.CODE_DIR / "pre-prompts" / pre_prompt_filename


    @config.debug_function
    def process_samples_llm(self, batch_size: int) -> list[str]:
        """
        Process the testing set in batches

        returns:
            the raw string outputs of the LLM
        """
        import torch

        outputs = []

        input_queue = mp.Queue()
        output_queue = mp.Queue()

        config.debug("Creating process")
        p = mp.Process(target=self.create_batch_process_worker,
                       args=(input_queue, output_queue, self.pre_prompt_path))
        p.start()

        batch_start = 0
        N = len(self.X_test)

        while batch_start < N:
            batch = self.X_test[batch_start : min(batch_start + batch_size, N)]
            batch_number = batch_start // batch_size
            batch_count = N // batch_size

            config.debug(f"Processing batch {batch_number} of {batch_count}")

            input_queue.put(batch)
            results = output_queue.get()

            if results["successful"]:
                outputs.extend(results["output"])
                batch_start += batch_size
            else:
                config.debug("Killing process and reloading model")

                if p.is_alive():
                    p.terminate()
                    p.join()

                torch.cuda.empty_cache()

                # Create new queues
                input_queue = mp.Queue()
                output_queue = mp.Queue()

                config.debug("Creating process")
                p = mp.Process(target=self.create_batch_process_worker,
                                args=(input_queue, output_queue, self.pre_prompt_path))
                p.start()

                new_batch_size = max(1, int(0.8 * batch_size))

                if new_batch_size == batch_size > 1:
                    new_batch_size -= 1

                config.debug(f"Reduced batch size from {batch_size} to {new_batch_size}")

                batch_size = new_batch_size

        input_queue.put(None)
        p.join()

        return outputs


    @classmethod
    def create_batch_process_worker(cls, input_queue: mp.Queue, output_queue: mp.Queue, pp_path: Path):
        """
        Creates a batch processing worker

        args:
        - input_queue: the queue this process take batches from
        - output_queue: the queue this process writes output to, as dicts:
            - "successful": whether the processing was successful (or OOM)
            - "output": the list of text output from the model
        - pp_path: pre-prompt path
        """
        import torch

        m = model.BatchModel(config.MODEL)
        m.load_pre_prompt(pp_path)

        with torch.no_grad():
            while True:
                if (batch := input_queue.get()) is None:
                    break

                try:
                    output = m.process_batch(batch, enforce_json=True)
                    output_queue.put({ "successful": True, "output": output })
                except RuntimeError as e:
                    if str(e).startswith('CUDA out of memory'):
                        output_queue.put({ "successful": False, "output": None })
                    else:
                        raise e

        torch.cuda.empty_cache()


    def run_experiment(self, batch_size: int) -> Tuple[pd.DataFrame, list[str]]:
        """
        Runs the experiment with a set batch size

        returns:
            A pd.DataFrame with columns:
                - valid_json: whether the output made by the LLM was valid
                - true_age: the correct age
                - true_gender: the correct gender
                - pred_age: predicted age
                - pred_gender: predicted gender
            as well as the original output strings
        """
        config.debug("Running experiment")
        y_test_outputs = self.process_samples_llm(batch_size)
        y_pred = self.extract_json(y_test_outputs, {"age": None, "gender": None})

        df_true = pd.DataFrame.from_records(self.y_test).rename(columns={"age": "true_age", "gender": "true_gender"})
        df_pred = pd.DataFrame.from_records(y_pred).rename(columns={"age": "pred_age", "gender": "pred_gender"})

        df = pd.concat([df_true, df_pred], axis=1)

        df["valid_json"] = ~df["pred_age"].isna()
        df["pred_age"] = pd.to_numeric(df_pred["pred_age"], errors="coerce")

        return df, y_test_outputs


    @classmethod
    @config.debug_function
    def extract_json(cls, outputs: list[str], default: dict) -> list[dict]:
        """ 
        Attempts to extract and parse valid json from each output string

        args:
        - outputs: string outputs to parse
        - default: default object to use if not parseable

        For each output that doesn't yield valid output, None is put in its place
        """

        json_outputs = []

        for s in outputs:
            try:
                potential_json = s.split("}")[0] + "}"
                parsed = json.loads(potential_json)
            except json.JSONDecodeError:
                parsed = default.copy()

            json_outputs.append(parsed)

        return json_outputs


    def evaluation(self, results: pd.DataFrame) -> dict:
        """
        Evaluates the results of the LLM inference

        args:
        - results: this is a pandas dataframe with columns:
            - valid_json: whether the output made by the LLM was valid
            - true_age: the correct age
            - true_gender: the correct gender
            - pred_age: predicted age
            - pred_gender: predicted gender

        returns:
        A dict containing:
        - an evaluation of the prediction performance on gender
        - "" "" "" age prediction
        """
        valid = results[results["valid_json"] == True].copy()
        num_total = len(results)
        num_valid = len(valid)

        config.debug(f"Total samples: {num_total}")
        config.debug(f"Valid JSON outputs: {num_valid} ({100 * num_valid / num_total:.2f}%)")

        # gender
        gender_mask = valid["true_gender"].isin(["male", "female"]) & valid["pred_gender"].isin(["male", "female"])
        gender_df = valid[gender_mask]

        if not gender_df.empty:
            gender_accuracy = accuracy_score(gender_df["true_gender"], gender_df["pred_gender"])
            gender_precision = precision_score(gender_df["true_gender"], gender_df["pred_gender"], pos_label="female", average="binary")
            gender_recall = recall_score(gender_df["true_gender"], gender_df["pred_gender"], pos_label="female", average="binary")
            gender_f1 = f1_score(gender_df["true_gender"], gender_df["pred_gender"], pos_label="female", average="binary")

            config.debug(f"Gender accuracy: {gender_accuracy:.3f}")
            config.debug(f"Gender precision: {gender_precision:.3f}")
            config.debug(f"Gender recall: {gender_recall:.3f}")
            config.debug(f"Gender F1-score: {gender_f1:.3f}")
        else:
            gender_accuracy = gender_precision = gender_recall = gender_f1 = None
            config.debug("No valid gender rows (male/female) for evaluation.")

        # age metrics
        pred_age = pd.to_numeric(valid["pred_age"], errors="coerce")
        true_age = pd.to_numeric(valid["true_age"], errors="coerce")

        # Filter NaNs caused by conversion issues
        mask = pred_age.notna() & true_age.notna()
        pred_age = pred_age[mask]
        true_age = true_age[mask]

        if not pred_age.empty:
            age_mae = mean_absolute_error(true_age, pred_age)
            age_rmse = root_mean_squared_error(true_age, pred_age)
            within_3_years = np.mean(np.abs(true_age - pred_age) <= 3)

            config.debug(f"Age MAE: {age_mae:.2f}")
            config.debug(f"Age RMSE: {age_rmse:.2f}")
            config.debug(f"Accuracy within ±3 years: {within_3_years:.2%}")
        else:
            age_mae = age_rmse = within_3_years = None
            config.debug("No valid age rows for evaluation.")

        return {
            "num_total": num_total,
            "num_valid_json": num_valid,
            "gender": {
                "accuracy": gender_accuracy,
                "precision": gender_precision,
                "recall": gender_recall,
                "f1": gender_f1
            },
            "age": {
                "mae": age_mae,
                "rmse": age_rmse,
                "accuracy_within_3_years": within_3_years
            }
        }


if __name__ == "__main__":
    mp.set_start_method("spawn")

    expt1 = Experiment1("expt1-zero-shot.txt", num_samples=10_000)

    expt_results, expt_output = expt1.run_experiment(20)
    expt_evaluation = expt1.evaluation(expt_results)

    with open(config.RESULTS_DIR / "experiment-1-output.json", "w") as f:
        json.dump(expt_output, f, indent=2)

    with open(config.RESULTS_DIR / "experiment-1-evaluation.json", "w") as f:
        json.dump(expt_evaluation, f, indent=2)

    expt_results.to_parquet(config.RESULTS_DIR / "experiment-1-parsed-output.parquet", index=False)
