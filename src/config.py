import datetime
import os
from pathlib import Path


RELEVANT_SUBREDDITS = [
    "food", "soda", "alcohol", "fastfood", "australia", "europe", "ireland",
    "news", "politics", "sweden", "unitedkingdom", "worldnews", "keto",
    "loseit", "nutrition", "running", "vegan", "vegetarian", "AskReddit",
    "LifeProTips", "technology", "tifu", "todayilearned", "science"
]

# control
IRRELEVANT_SUBREDDITS = [
    "leagueoflegends", "pcmasterrace", "StarWars", "Android", "legaladvice"
]


DEBUG = True
SMALL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
#LARGE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
LARGE_MODEL = SMALL_MODEL

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

def output(msg: str, msg_type: str = "output"):
    """
    Prints a message with a timestamp 
    """
    print(f"[{msg_type} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {msg}")


def debug(msg: str):
    """
    Prints the debug message, if DEBUG == True
    """

    global DEBUG

    if DEBUG:
        output(msg, msg_type="debug")


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