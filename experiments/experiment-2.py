import itertools
import json
import numpy as np
import os
import sys
import torch

# Set path for local imports
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

import config
import dataset
import model


if __name__ == "__main__":
    config.debug(f"CUDA is available: {torch.cuda.is_available()}")
    config.debug(f"CUDA device 0 name: {torch.cuda.get_device_name(0)}")
