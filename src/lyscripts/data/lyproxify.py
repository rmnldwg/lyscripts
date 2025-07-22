"""Consumes raw data and transforms it into a CSV that `LyProX`_ understands.

To do so, it needs a dictionary that defines a mapping from raw columns to the LyProX
style data format. See the documentation of the :py:func:`.transform_to_lyprox` function
for more information.

.. _LyProX: https://lyprox.org
"""

import importlib.util
import warnings
from pathlib import Path
from typing import Annotated, Any

import lydata  # noqa: F401
import pandas as pd
from loguru import logger
from lydata import C
from pydantic import AfterValidator, Field, FilePath

from lyscripts.cli import assemble_main
from lyscripts.configs import BaseCLI
from lyscripts.data.utils import save_table_to_csv
from lyscripts.utils import delete_private_keys, flatten, load_patient_data

warnings.simplefilter(action="ignore", category=FutureWarning)


def ensure_python_file(file: Path) -> Path:
    """Check if the file is a Python file."""
    if file.suffix != ".py":
        raise ValueError("Mapping file must be a Python file.")

    return file


def ensure_column_map(file: Path) -> Path:
    """Ensure the Python file contains a ``COLUMN_MAP`` dictionary."""
    spec = importlib.util.spec_from_file_location("map_module", file)
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)

    if not hasattr(mapping, "COLUMN_MAP"):
        raise ValueError("Mapping file must contain a `COLUMN_MAP` dictionary.")

    return file


class LyproxifyCLI(BaseCLI):
    """Map any CSV file to the LyProX format with the help of a Python mapping dict."""

    input_file: FilePath = Field(description="Location of raw CSV data.")
    num_header_rows: int = Field(
        default=1,
        description="Number of rows comprising the header of the raw CSV file.",
    )
    mapping_file: Annotated[
        FilePath,
        AfterValidator(ensure_python_file),
        AfterValidator(ensure_column_map),
    ] = Field(
        description=(
            "Location of Python file containing a `COLUMN_MAP` dictionary. It may also "
            "contain an `EXCLUDE` list of tuples `(column, check)` to exclude patients."
        ),
    )
    drop_rows: list[int] = Field(
        default=[],
        description=(
            "Delete rows of specified indices. Counting of rows start at 0 _after_ "
            "the `header-rows`."
        ),
    )
    drop_cols: list[int] = Field(
        default=[],
        description="Delete columns of specified indices.",
    )
    output_file: Path = Field(description="Location to store the lyproxified CSV file.")

    def cli_cmd(self) -> None:
        """Start the ``lyproxify`` subcommand.

        After reading in the specified file, it will first ``drop_rows`` and
        ``drop_cols``, as specified in the command line arguments. Then, it will
        call :py:func:`.exclude_patients` which will further remove patients based
        on the ``EXCLUDE`` object in the ``mapping_file``. Finally, it will call
        :py:func:`.transform_to_lyprox` to transform the data into the LyProX format
        given the ``COLUMN_MAP`` object in the ``mapping_file``.
        """
        logger.debug(self.model_dump_json(indent=2))

        raw = load_patient_data(
            file_path=self.input_file,
            header=list(range(self.num_header_rows)),
        )
        raw = clean_header(
            table=raw,
            num_cols=raw.shape[1],
            num_header_rows=self.num_header_rows,
        )

        cols_to_drop = raw.columns[self.drop_cols]
        trimmed = raw.drop(cols_to_drop, axis="columns")
        trimmed = trimmed.drop(index=self.drop_rows)
        trimmed = trimmed.dropna(axis="index", how="all")
        logger.info(f"Dropped rows {self.drop_rows} and columns {cols_to_drop}.")

        spec = importlib.util.spec_from_file_location("map_module", self.mapping_file)
        mapping = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapping)
        logger.info(f"Imported mapping instructions from {self.mapping_file}")

        reduced = exclude_patients(trimmed, mapping.EXCLUDE)
        processed = transform_to_lyprox(reduced, mapping.COLUMN_MAP)

        if "side" in processed.ly:
            processed = leftright_to_ipsicontra(processed)

        save_table_to_csv(file_path=self.output_file, table=processed)


class ParsingError(Exception):
    """Error while parsing the CSV file."""


def clean_header(
    table: pd.DataFrame,
    num_cols: int,
    num_header_rows: int,
) -> pd.DataFrame:
    """Rename the header cells in the ``table``."""
    table = table.copy()

    for col in range(num_cols):
        for row in range(num_header_rows):
            table.rename(
                columns={f"Unnamed: {col}_level_{row}": f"{col}_lvl_{row}"},
                inplace=True,
            )

    logger.debug("Cleaned headers of the raw data.")
    return table


def get_instruction_depth(nested_column_map: dict[tuple, dict[str, Any]]) -> int:
    """Get the depth at which the column mapping instructions are nested.

    Instructions are a dictionary that contains either a 'func' or 'default' key.

    >>> nested_column_map = {"patient": {"age": {"func": int}}}
    >>> get_instruction_depth(nested_column_map)
    2
    >>> flat_column_map = flatten(nested_column_map, max_depth=2)
    >>> get_instruction_depth(flat_column_map)
    1
    >>> nested_column_map = {"patient": {"__doc__": "some patient info", "age": 61}}
    >>> get_instruction_depth(nested_column_map)
    Traceback (most recent call last):
        ...
    ValueError: Leaf of column map must be a dictionary with 'func' or 'default' key.
    """
    for _, value in nested_column_map.items():
        if isinstance(value, dict):
            if "func" in value or "default" in value:
                return 1

            return 1 + get_instruction_depth(value)

        raise ValueError(
            "Leaf of column map must be a dictionary with 'func' or 'default' key.",
        )

    raise ValueError("Empty column map.")


def generate_markdown_docs(
    nested_column_map: dict[tuple, dict[str, Any]],
    depth: int = 0,
    indent_len: int = 4,
) -> str:
    r"""Generate a markdown nested, ordered list as documentation for the column map.

    A key in the doctionary is supposed to be documented, when its value is a dictionary
    containing a ``"__doc__"`` key.

    >>> nested_column_map = {
    ...     "patient": {
    ...         "__doc__": "some patient info",
    ...         "age": {
    ...             "__doc__": "age of the patient",
    ...             "func": int,
    ...             "columns": ["age"],
    ...         },
    ...     },
    ... }
    >>> generate_markdown_docs(nested_column_map)
    '1. **`patient:`** some patient info\n    1. **`age:`** age of the patient\n'
    """
    md_docs = ""
    indent = " " * indent_len * depth
    i = 1
    for key, value in nested_column_map.items():
        if isinstance(value, dict):
            if "__doc__" in value:
                md_docs += f"{indent}{i}. **`{key}:`** {value['__doc__']}\n"
                i += 1

            md_docs += generate_markdown_docs(value, depth + 1, indent_len)

    return md_docs


def transform_to_lyprox(
    raw: pd.DataFrame,
    column_map: dict[tuple, dict[str, Any]],
) -> pd.DataFrame:
    """Transform ``raw`` data into table that can be uploaded directly to `LyProX`_.

    To do so, it uses instructions in the `colum_map` dictionary, that needs to have
    a particular structure:

    For each column in the final 'lyproxified' `pd.DataFrame`, one entry must exist in
    the `column_map` dictionary. E.g., for the column corresponding to a patient's age,
    the dictionary should contain a key-value pair of this shape:

    .. code-block:: python

        column_map = {
            ("patient", "info", "age"): {
                "func": compute_age_from_raw,
                "kwargs": {"randomize": False},
                "columns": ["birthday", "date of diagnosis"]
            },
        }

    In this example, the function ``compute_age_from_raw`` is called with the
    values of the columns ``"birthday"`` and ``"date of diagnosis"`` as positional
    arguments, and the keyword argument ``"randomize"`` is set to ``False``. The
    function then returns the patient's age, which is subsequently stored in the column
    ``("patient", "info", "age")``.

    Note that the ``column_map`` dictionary must have either a ``"default"`` key or
    ``"func"`` along with ``"columns"`` and ``"kwargs"``, depending on the function
    definition. If the function does not take any arguments, ``"columns"`` can be
    omitted. If it also does not take any keyword arguments, ``"kwargs"`` can be
    omitted, too.

    .. _LyProX: https://lyprox.org
    """
    column_map = delete_private_keys(column_map)

    if (instruction_depth := get_instruction_depth(column_map)) > 1:
        column_map = flatten(column_map, max_depth=instruction_depth)

    multi_idx = pd.MultiIndex.from_tuples(column_map.keys())
    processed = pd.DataFrame(columns=multi_idx)

    for multi_idx_col, instruction in column_map.items():
        if instruction != "":
            if "default" in instruction:
                processed[multi_idx_col] = [instruction["default"]] * len(raw)
            elif "func" in instruction:
                cols = instruction.get("columns", [])
                kwargs = instruction.get("kwargs", {})
                func = instruction["func"]

                try:
                    processed[multi_idx_col] = [
                        func(*vals, **kwargs) for vals in raw[cols].values
                    ]
                except Exception as exc:
                    raise ParsingError(
                        f"Exception encountered while parsing column {multi_idx_col}",
                    ) from exc
            else:
                raise ParsingError(
                    f"Column {multi_idx_col} has neither a `default` value nor `func` "
                    "describing how to fill this column.",
                )

    logger.info("Transformed raw data to LyProX format.")
    return processed


def leftright_to_ipsicontra(data: pd.DataFrame):
    """Change absolute side reporting to tumor-relative.

    Transform reporting of LNL involvement by absolute side (right & left) to a
    reporting relative to the tumor (ipsi- & contralateral). The table ``data`` should
    already be in the format LyProX requires, except for the side-reporting of LNL
    involvement.
    """
    len_before = len(data)
    left_data = data.ly.query(C("side") != "right")
    right_data = data.ly.query(C("side") == "right")

    left_data = left_data.rename(columns={"left": "ipsi"}, level=1)
    left_data = left_data.rename(columns={"right": "contra"}, level=1)
    right_data = right_data.rename(columns={"left": "contra"}, level=1)
    right_data = right_data.rename(columns={"right": "ipsi"}, level=1)

    data = pd.concat([left_data, right_data], ignore_index=True)
    if len_before != len(data):
        raise RuntimeError("Number of patients changed")

    logger.info("Transformed side reporting to ipsi- and contralateral.")
    return data


def exclude_patients(raw: pd.DataFrame, exclude: list[tuple[str, Any]]):
    """Exclude patients in the ``raw`` data based on a list of what to ``exclude``.

    The ``exclude`` list contains tuples ``(column, check)``. The ``check`` function
    will then exclude any patients from the cohort where ``check(raw[column])``
    evaluates to ``True``.

    >>> exclude = [("age", lambda s: s > 50)]
    >>> table = pd.DataFrame({
    ...     "age":        [43, 82, 18, 67],
    ...     "T-category": [ 3,  4,  2,  1],
    ... })
    >>> exclude_patients(table, exclude)
       age  T-category
    0   43           3
    2   18           2
    """
    num_before = len(raw)
    filtered = raw.copy()

    for column, check in exclude:
        is_excluded = check(filtered[column])
        filtered = filtered.loc[~is_excluded]

    num_after = len(filtered)
    logger.info(f"Excluded {num_before - num_after} patients.")
    return filtered


if __name__ == "__main__":
    main = assemble_main(settings_cls=LyproxifyCLI, prog_name="lyproxify")
    main()
