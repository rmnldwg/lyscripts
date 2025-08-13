"""Submodule to collect data interactively using a simple web interface.

With the simply command

.. code-block:: bash

    lyscripts data collect

One can start a very basic web server that serves an interactive UI at
``http://localhost:8000/``. There, one can enter patient, tumor, and lymphatic
involvement data one by one. When completed, the "submit" button will parse, validate,
and convert the data to serve a downloadable CSV file.

This resulting CSV file is in the correct format to be used in `LyProX`_ and for
inference using our `lymph-model`_ library.

.. _LyProX: https://lyprox.org
.. _lymph-model: https://lymph-model.readthedocs.io
"""

import inspect
import io
import logging
from pathlib import Path
from typing import Any

import lydata
import lydata.validator
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import Field, RootModel
from starlette.responses import FileResponse, HTMLResponse

from lyscripts.cli import _current_log_level
from lyscripts.configs import BaseCLI

app = FastAPI(
    title="lyDATA Collector",
    description=(
        "A simple web interface to collect data for the lyDATA datasets. "
        "This is a prototype and not intended for production use."
    ),
    version=lydata.__version__,
)

BASE_DIR = Path(__file__).parent
modalities = lydata.schema.get_default_modalities()
RecordModel = lydata.schema.create_full_record_model(modalities, title="Record")
ROOT_MODEL = RootModel[list[RecordModel]]


@app.get("/")
def serve_index() -> HTMLResponse:
    """Serve the index.html file."""
    with open(BASE_DIR / "index.html") as file:
        content = file.read()
    return HTMLResponse(content=content)


@app.get("/schema")
def serve_schema() -> dict[str, Any]:
    """Serve the JSON schema for the patient and tumor records."""
    return ROOT_MODEL.model_json_schema()


@app.get("/collector.js")
def serve_collector_js() -> FileResponse:
    """Serve the collector.js file."""
    return FileResponse(BASE_DIR / "collector.js")


@app.post("/submit")
async def process(data: RootModel) -> StreamingResponse:
    """Convert the submitted data to a DataFrame."""
    logger.info(f"Received data: {data.root}")

    if len(data.root) == 0:
        logger.warning("No records provided in the data.")
        raise HTTPException(
            status_code=400,
            detail="No records provided in the data.",
        )

    flattened_records = []

    for record in data.root:
        flattened_record = lydata.validator.flatten(record)
        logger.debug(f"Flattened record: {flattened_record}")
        flattened_records.append(flattened_record)

    df = pd.DataFrame(flattened_records)
    df.columns = pd.MultiIndex.from_tuples(flattened_record.keys())
    logger.info(df.patient.core.head())

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    logger.success("Data prepared for download")
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=lydata_records.csv"},
    )


class InterceptHandler(logging.Handler):
    """Intercept logging messages and redirect them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Intercept the log record and redirect it to Loguru."""
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


class CollectorCLI(BaseCLI):
    """Command-line interface for the lyDATA collector."""

    hostname: str = Field(
        default="localhost",
        description="Hostname to run the FastAPI app on.",
    )
    port: int = Field(
        default=8000,
        description="Port to run the FastAPI app on.",
    )

    def cli_cmd(self) -> None:
        """Run the FastAPI app."""
        logger.debug(self.model_dump_json(indent=2))
        import uvicorn

        # Intercept standard logging and redirect it to Loguru
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        logger.enable("lydata")

        uvicorn.run(
            app,
            host=self.hostname,
            port=self.port,
            log_level=_current_log_level.lower(),
            log_config=None,
        )
