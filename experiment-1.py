import itertools
import json
import numpy as np
import torch

import config
import dataset
import model


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
    subset_offset = 100_000

    X_test = X[subset_offset : subset_offset + subset_size]
    y_test = y[subset_offset : subset_offset + subset_size]

    outputs = m.query_sequence_batched(X_test, batch_size=16)
    json_outputs = m.extract_json(outputs)

    for i in range(len(outputs)):
        y_true = y_test[i]
        y_pred = json_outputs[i]

        if y_pred:
            print(f"""instance {i}:\n
    - age:
        pred: {y_pred["age"]}
        true: {y_true["age"]}
    - gender:
        pred: {y_pred["gender"]}
        true: {y_true["gender"]}""")
        else:
            print(f"Regrettably this is the output: {outputs[i]}")
