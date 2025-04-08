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

    df["pred_age"] = pd.to_numeric(df_pred["pred_age"], errors="coerce")

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
    valid = results[results["valid_json"] == True].copy()
    num_total = len(results)
    num_valid = len(valid)

    config.debug(f"Total samples: {num_total}")
    config.debug(f"Valid JSON outputs: {num_valid} ({100 * num_valid / num_total:.2f}%)")

    # --- GENDER METRICS ---
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

    # --- AGE METRICS ---
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
    config.debug(f"CUDA is available: {torch.cuda.is_available()}")
    config.debug(f"CUDA device 0 name: {torch.cuda.get_device_name(0)}")

    config.debug("Loading training and validation sets")
    df_train, X_train, y_train = dataset.load_dataset_locally("blog_authorship_corpus",
                                                              "text",["age", "gender"])

    df_val, X_val, y_val = dataset.load_dataset_locally("blog_authorship_corpus",
                                                        "text", ["age", "gender"])

    config.debug("producing sample from validation set")
    num_samples = 10000
    np.random.seed(42)
    total_val = len(X_val)
    selected_indices = np.random.choice(np.arange(total_val), size=num_samples, replace=False)
    selected_indices.sort()

    with open(config.RESULTS_DIR / "subset-indices.txt", "w") as f:
        f.write("\n".join(map(str, selected_indices)))

    # Create validation subset
    X_subset = [X_val[i] for i in selected_indices]
    y_subset = [y_val[i] for i in selected_indices]

    m = model.SequenceModel(config.MODEL)
    m.load_pre_prompt(config.CODE_DIR / "pre-prompts" / "expt1-zero-shot.txt")

    # compute and write outputs
    with torch.no_grad():
        outputs = m.query_sequence_batched(X_subset, batch_size=10)

    # parse json
    json_outputs = m.extract_json(outputs)
    results = output_to_dataframe(json_outputs, y_subset)

    # Experiment evaluation
    results_evaluation = evaluation(results)

    # Write all output
    with open(config.RESULTS_DIR / "experiment-1-output.txt", "w") as f:
        f.write("\n".join(outputs))

    with open(config.RESULTS_DIR / "evaluation-experiment-1-output.json", "w") as f:
        json.dump(results_evaluation, f, indent=2)

    results.to_parquet(config.RESULTS_DIR / "parsed-experiment-1-output.parquet", index=False)
