import itertools
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import *
import sys
import torch

# Set path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import dataset
import model


def output_to_dataframe(y_pred: list[dict], y_true: list[dict]) -> pd.DataFrame:
    """
    Converts a list of JSON outputs to a pandas dataframe

    args:
    - y_pred: the predictions
    - y_true: the actual age and gender values

    returns:
    A pd.DataFrame with columns:
        - valid_json: whether the output made by the LLM was valid
        - true_age: the correct age
        - true_gender: the correct gender
        - pred_age: predicted age
        - pred_gender: predicted gender
    """
    df_true = pd.DataFrame.from_records(y_true).rename(columns={"age": "true_age", "gender": "true_gender"})
    df_pred = pd.DataFrame.from_records(y_pred).rename(columns={"age": "pred_age", "gender": "pred_gender"})

    df = pd.concat([df_true, df_pred], axis=1)
    df["valid_json"] = ~df["pred_age"].isna()

    return df


def evaluation(results: pd.DataFrame) -> dict:
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
    valid = results[results["valid_json"] != True].copy()
    num_total = len(results)
    num_valid = len(valid)

    config.debug(f"Total samples: {num_total}")
    config.debug(f"Valid JSON outputs: {num_valid} ({100 * num_valid / num_total:.2f}%)")

    # --- GENDER METRICS ---
    gender_accuracy = accuracy_score(valid["true_gender"], valid["pred_gender"])
    gender_precision = precision_score(valid["true_gender"], valid["pred_gender"], pos_label="female", average="binary")
    gender_recall = recall_score(valid["true_gender"], valid["pred_gender"], pos_label="female", average="binary")
    gender_f1 = f1_score(valid["true_gender"], valid["pred_gender"], pos_label="female", average="binary")

    config.debug(f"Gender accuracy: {gender_accuracy:.3f}")
    config.debug(f"Gender precision: {gender_precision:.3f}")
    config.debug(f"Gender recall: {gender_recall:.3f}")
    config.debug(f"Gender F1-score: {gender_f1:.3f}")

    # --- AGE METRICS ---
    pred_age = pd.to_numeric(valid["pred_age"], errors="coerce")
    true_age = pd.to_numeric(valid["true_age"], errors="coerce")

    # Filter NaNs caused by conversion issues
    mask = pred_age.notna() & true_age.notna()
    pred_age = pred_age[mask]
    true_age = true_age[mask]

    age_mae = mean_absolute_error(true_age, pred_age)
    age_rmse = mean_squared_error(true_age, pred_age, squared=False)
    within_3_years = np.mean(np.abs(true_age - pred_age) <= 3)

    config.debug(f"Age MAE: {age_mae:.2f}")
    config.debug(f"Age RMSE: {age_rmse:.2f}")
    config.debug(f"Accuracy within ±3 years: {within_3_years:.2%}")

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
    config.debug(f"CUDA is available: {torch.cuda.is_available()}")
    config.debug(f"CUDA device 0 name: {torch.cuda.get_device_name(0)}")

    df, X, y = dataset.load_dataset_locally("blog_authorship_corpus", "text", ["age", "gender"])
    m = model.SequenceModel(config.MODEL)

    m.set_pre_prompt("""You are an AI assistant which:
1. Only receives un-annotated social media posts as input
2. Only emits output in valid JSON format, with this schema:
    {
        "age": <your prediction of the age (**strictly** an integer)>,
        "gender": <your prediction of the gender of the poster, either 'male' or 'female'>
    }

You will now receive a single input, and you must reply **only** in JSON, with no extra text.
If you are unsure of the classification for age or gender, guess.""")

    subset_size = 1000
    subset_offset = 100_100

    X_test = X[subset_offset : subset_offset + subset_size]
    y_test = y[subset_offset : subset_offset + subset_size]

    # compute and write outputs
    outputs = m.query_sequence_batched(X_test, batch_size=32)

    with open(config.RESULTS_DIR / "experiment-1-output.txt", "w") as f:
        f.write("\n".join(outputs))

    # parse json
    json_outputs = m.extract_json(outputs)
    results = output_to_dataframe(json_outputs, y_test)

    # Experiment evaluation
    results_evaluation = evaluation(results)

    # Write all output
    with open(config.RESULTS_DIR / "experiment-1-output.txt", "w") as f:
        f.write("\n".join(outputs))

    with open(config.RESULTS_DIR / "evaluation-experiment-1-output.json", "w") as f:
        json.dump(results, f, indent=2)

    results.to_parquet(config.RESULTS_DIR / "parsed-experiment-1-output.parquet", index=False)
