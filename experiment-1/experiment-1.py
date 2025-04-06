import itertools
import json
import numpy as np
import os
import pandas as pd
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
    N = len(outputs)

    valid_json = np.full(True, size=N)
    true_age = np.full(np.nan, size=N)
    true_gender = np.full("", size=N)
    pred_age = np.full(np.nan, size=N)
    pred_gender = np.full("", size=N)

    for i, (p, t) in enumerate(zip(y_pred, y_true)):
        true_age[i] = t["age"]
        true_gender[i] = t["gender"]

        if p:
            pred_age[i] = p["age"]
            pred_gender[i] = p["gender"]
        else:
            valid_json[i] = False

    return pd.DataFrame({
        "valid_json": valid_json,
        "true_age": true_age,
        "true_gender": true_gender,
        "pred_age": pred_age,
        "pred_gender": pred_gender
    })
    


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
    subset_offset = 100_000

    X_test = X[subset_offset : subset_offset + subset_size]
    y_test = y[subset_offset : subset_offset + subset_size]

    outputs = m.query_sequence_batched(X_test, batch_size=8)
    json_outputs = m.extract_json(outputs)

    results = output_to_dataframe(json_outputs, y_test)

    results.to_parquet("/project/experiment-1/output.parquet", index=False)
