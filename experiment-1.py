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

    print(f"X[0] = {X[0]}")
