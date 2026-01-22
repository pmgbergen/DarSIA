from time import time
from functools import wraps

import logging

logger = logging.getLogger(__name__)


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        # Do not only use the name of the function, but also the class it belongs to
        logger.info(
            f"{func.__module__}.{func.__name__} executed in {end - start:.3f} seconds"
        )
        return result

    return wrapper
