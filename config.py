import datetime
import os
from pathlib import Path


DEBUG = True
SMALL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# directories
CACHE_DIR = Path.home() / ".cache"
DATASET_DIR = Path.home() / "data"
TMP_DIR = Path.home() / ".tmp"
RESULTS_DIR = Path.home() / "results"
CODE_DIR = Path.home() / "final-year-project"

# specific to selecting posts
ACCEPTED_SUBREDDITS = [
    # Relating to the core business
    "food",
    "soda",
    "alcohol",
    "fastfood",

    # Relating to current events and locations (english language)
    "australia",
    "europe",
    "ireland",
    "news",
    "politics",
    "sweden",
    "unitedkingdom",
    "worldnews",

    # Relating to fitness or diet
    "fitness",
    "keto",
    "loseit",
    "nutrition",
    "running",
    "vegan",
    "vegetarian",

    # Popular reddit forums
    "AskReddit",
    "LifeProTips",
    "technology",
    "tifu",
    "todayilearned",
    "science",
]


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