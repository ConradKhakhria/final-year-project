import config
import dataset
import sys


if __name__ == "__main__":
  dataset_name = sys.argv[1]
  
  config.debug("Starting dataset caching")
  df, X, y = dataset.load_data_locally(dataset_name, "text", ["age", "gender"])
  config.debug(f"X.shape = {X.shape}")
