CHECK_CUDA = True
DEBUG = True
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


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
