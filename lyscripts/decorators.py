"""Decorators to avoid repetitive snippets of code.

E.g. safely opening files or logging the state of a function call.

This is *not* a command line tool.
"""

import functools
import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any


def assemble_signature(*args, **kwargs) -> str:
    """Assemble the signature of the function call."""
    args_str = ", ".join(str(arg) for arg in args)
    kwargs_str = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    return ", ".join([args_str, kwargs_str])


def log_state(log_level: int = logging.INFO) -> Callable:
    """Provide a decorator that logs the state of the function execution.

    The log message will simply be the function name where underscores are replaced
    with spaces. The `log_level` can be set in the decorator call.
    """

    def log_decorator(func: Callable):
        """Decorate function for which to add logs."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute decorated function."""
            logger = logging.getLogger(func.__module__)
            signature = assemble_signature(*args, **kwargs)
            logger.debug(f"Executing {func.__name__}({signature}).")
            log_msg_from_func = func.__name__.replace("_", " ").capitalize() + "."

            try:
                logger.log(
                    log_level,
                    log_msg_from_func,
                    extra={
                        "func_filepath": f"{func.__module__.replace('.', '/')}.py",
                        "func_name": func.__name__,
                        "module_name": func.__module__,
                    },
                )
                return func(*args, **kwargs)

            except Exception as exc:
                logger.error(f"Error calling {func.__name__}().", exc_info=exc)
                raise exc

        return wrapper

    return log_decorator


def check_input_file_exists(loading_func: Callable) -> Callable:
    """Check if the file path provided to the `loading_func` exists."""

    @wraps(loading_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Execute wrapped loading function."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        return loading_func(file_path, *args, **kwargs)

    return inner


def check_output_dir_exists(saving_func: Callable) -> Callable:
    """Make sure the parent directory of the saved file exists."""

    @wraps(saving_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Execute wrapped saving function."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return saving_func(file_path, *args, **kwargs)

    return inner
