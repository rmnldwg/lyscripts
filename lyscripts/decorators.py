"""
This module provides decorators that can be used to avoid repetitive snippets of code,
e.g. safely opening files or logging the state of a function call.
"""
import functools
import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO, TextIO


def extract_logger(*args, **kwargs) -> logging.Logger:
    """Extract a logger from the arguments if present.

    The function will first search if the function is in fact a method of a class that
    has any attributes that are instances of `logging.Logger`. If not, it will search
    the arguments and keyword arguments for instances of `logging.Logger` and return
    the first one it finds. If none is found, it will return a general logger.
    """
    first_arg = next(iter(args), None)
    attr_loggers = []
    if hasattr(first_arg, "__dict__"):
        for attr in first_arg.__dict__.values():
            if isinstance(attr, logging.Logger):
                attr_loggers.append(attr)

    return_args = []
    args_loggers = []
    for arg in args:
        if isinstance(arg, logging.Logger):
            args_loggers.append(arg)
        else:
            return_args.append(arg)

    return_kwargs = {}
    kwargs_loggers = []
    for key, value in kwargs.items():
        if isinstance(value, logging.Logger):
            kwargs_loggers.append(value)
        else:
            return_kwargs[key] = value

    found_loggers = [*attr_loggers, *args_loggers, *kwargs_loggers]
    logger = next(iter(found_loggers), None)

    if logger is None:
        logger = logging.getLogger("lyscripts")

    return logger, return_args, return_kwargs


def assemble_signature(*args, **kwargs) -> str:
    """Assemble the signature of the function call."""
    args_str = ", ".join(str(arg) for arg in args)
    kwargs_str = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    signature = ", ".join([args_str, kwargs_str])
    return signature


def log_state(log_level: int = logging.INFO) -> Callable:
    """Provide a decorator that logs the state of the function execution.

    The log message will simply be the function name where underscores are replaced
    with spaces. The `log_level` can be set in the decorator call.
    """
    # pylint: disable=logging-fstring-interpolation
    # pylint: disable=logging-not-lazy
    def log_decorator(func: Callable):
        """The decorator wrapping the decorated function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """The wrapper around the decorated function."""
            logger = logging.getLogger(func.__module__)
            signature = assemble_signature(*args, **kwargs)
            logger.debug(f"Executing {func.__name__}({signature}).")
            log_msg_from_func = func.__name__.replace("_", " ").capitalize() + "."

            try:
                result = func(*args, **kwargs)
                logger.log(log_level, log_msg_from_func)
                return result

            except Exception as exc:
                logger.error(f"Error calling {func.__name__}().", exc_info=exc)
                raise exc

        return wrapper

    return log_decorator


def check_input_file_exists(loading_func: Callable) -> Callable:
    """Check if the file path provided to the `loading_func` exists."""
    @wraps(loading_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped loading function."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        return loading_func(file_path, *args, **kwargs)

    return inner


def provide_file(is_binary: bool) -> Callable:
    """Make sure a decorated function is provided with a file-like object.

    This means, the assembled decorator checks the argument type and, if necessary,
    opens the file to call the decorated function. The provided file is either a text
    file of - if `is_binary` is set to `True` - a binary file.
    """
    def assembled_decorator(loading_func: Callable) -> Callable:
        """Assembled decorator that provides the function with a text/binary file."""
        @wraps(loading_func)
        def inner(file_or_path: str | Path | TextIO | BinaryIO, *args, **kwargs):
            """The wrapped function."""
            if isinstance(file_or_path, (str, Path)):
                file_path = Path(file_or_path)
                if not file_path.is_file():
                    raise FileNotFoundError(f"File {file_path} does not exist.")

                if is_binary:
                    with open(file_path, mode="rb") as bin_file:
                        return loading_func(bin_file, *args, **kwargs)
                else:
                    with open(file_path, encoding="utf-8") as txt_file:
                        return loading_func(txt_file, *args, **kwargs)

            return loading_func(file_or_path, *args, **kwargs)

        return inner

    return assembled_decorator


def check_output_dir_exists(saving_func: Callable) -> Callable:
    """Make sure the parent directory of the saved file exists."""
    @wraps(saving_func)
    def inner(file_path: str, *args, **kwargs) -> Any:
        """Wrapped saving function."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return saving_func(file_path, *args, **kwargs)

    return inner
