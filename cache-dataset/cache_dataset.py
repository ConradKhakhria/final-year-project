import os
import sys

current_dir = os.path.dirname(os.path.dirname(__file__), "..", "experiment-1")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
import dataset


if __name__ == "__main__":
  dataset_name = sys.argv[1]
  
  config.debug("Starting dataset caching")
  df, X, y = dataset.load_data_locally(dataset_name, "text", ["age", "gender"])
  config.debug(f"X.shape = {X.shape}")
