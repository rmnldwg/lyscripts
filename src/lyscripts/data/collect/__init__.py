"""Submodule to collect data interactively using a simple web interface.

With the simple command

.. code-block:: bash

    lyscripts data collect

One can start a very basic web server that serves an interactive UI at
``http://localhost:8000/``. There, one can enter patient, tumor, and lymphatic
involvement data one by one. When completed, the "submit" button will parse, validate,
and convert the data to serve a downloadable CSV file.

The resulting CSV file is in the correct format to be used in `LyProX`_ and for
inference using our `lymph-model`_ library.

.. _LyProX: https://lyprox.org
.. _lymph-model: https://lymph-model.readthedocs.io
"""

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

from lyscripts.cli import InterceptHandler, _current_log_level
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
RecordModel = lydata.schema.create_full_record_model(modalities, model_name="Record")
ROOT_MODEL = RootModel[list[RecordModel]]


@app.get("/")
def serve_index_html() -> HTMLResponse:
    """Serve the ``index.html`` file at the URL's root."""
    with open(BASE_DIR / "index.html") as file:
        content = file.read()
    return HTMLResponse(content=content)


@app.get("/schema")
def serve_schema() -> dict[str, Any]:
    """Serve the JSON schema for the patient and tumor records."""
    return ROOT_MODEL.model_json_schema()


@app.get("/collector.js")
def serve_collector_js() -> FileResponse:
    """Serve the ``collector.js`` file under ``"http://{host}:{port}/collector.js"``.

    This frontend JavaScript file loads the `JSON-Editor`_ library and initializes it
    using the schema returned by the :py:func:`serve_schema` function.

    .. _JSON-Editor: https://github.com/json-editor/json-editor/
    """
    return FileResponse(BASE_DIR / "collector.js")


@app.post("/submit")
async def process(data: RootModel) -> StreamingResponse:
    """Process the submitted data to a DataFrame.

    `FastAPI`_ will automatically parse the received JSON data into the list of
    instances of he pydantic type defined by the
    :py:func:`lydata.schema.create_full_record_model` function.

    From this list, we create a pandas DataFrame and return it as a downloadable CSV
    file.

    .. _FastAPI: https://fastapi.tiangolo.com/
    """
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


class CollectorCLI(BaseCLI):
    """Serve a FastAPI web app for collecting involvement patterns as CSV files."""

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
