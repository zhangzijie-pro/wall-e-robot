import logging
import functools
import os

class CustomLogger(logging.Logger):
    def findCaller(self, stack_info=False, stacklevel=1):
        f = logging.currentframe()
        for _ in range(stacklevel):
            if f is not None:
                f = f.f_back
        while f:
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if "logging" not in filename:
                return (filename, f.f_lineno, co.co_name, None)
            f = f.f_back
        return ("<unknown>", 0, "<unknown>", None)

def logger_config(name="custom_logger", log_file="speaker_embedding.log"):
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = logger_config()

def log():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
