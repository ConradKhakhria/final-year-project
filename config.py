import datetime
import os
from pathlib import Path


DEBUG = True
MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"

# directories
CACHE_DIR = Path.home() / ".cache"
DATASET_DIR = Path.home() / "data"
TMP_DIR = Path.home() / ".tmp"
RESULTS_DIR = Path.home() / "results"
CODE_DIR = Path.home() / "final-year-project"

# set env variables
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface-models")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Some helper functions

def debug(msg: str):
    """
    Prints the debug message, if DEBUG == True
    """

    global DEBUG

    if DEBUG:
        print(f"[debug {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {msg}")


def debug_function(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = getattr(func, '__name__', str(func))
        debug(f"calling {name}")
        result = func(*args, **kwargs)
        debug(f"{name} completed")
        return result

    return wrapper