import itertools
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
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
    df["valid_json"] = df["pred_age"].isna()

    return df


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

    subset_size = 100
    subset_offset = 100_100

    X_test = X[subset_offset : subset_offset + subset_size]
    y_test = y[subset_offset : subset_offset + subset_size]

    # compute and write outputs
    outputs = m.query_sequence_batched(X_test, batch_size=16)

    with open(config.RESULTS_DIR / "experiment-1-out.txt", "w") as f:
        f.write("\n".join(outputs))

    # parse json
    json_outputs = m.extract_json(outputs)
    results = output_to_dataframe(json_outputs, y_test)

    results.to_parquet(config.RESULTS_DIR / "experiment-1-output.parquet", index=False)
