from pathlib import Path


DEBUG = True
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# directories
LT_STORAGE_DIR = Path("/project")
LT_TEMPORARY_DIR = Path.home() / "Scratch" / "fyp"


# Some helper functions

def debug(msg: str):
    """
    Prints the debug message, if DEBUG == True
    """

    global DEBUG

    if DEBUG:
        print(f"[debug]: {msg}")


def debug_function(func):
    def wrapper(*args, **kwargs):
        debug(f"calling {func.__name__}")

        return func(*args, **kwargs)

    return wrapper
