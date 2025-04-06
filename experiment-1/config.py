from pathlib import Path


DEBUG = True
#MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

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
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = getattr(func, '__name__', str(func))
        debug(f"calling {name}")
        result = func(*args, **kwargs)
        debug(f"{name} completed")
        return result

    return wrapper