"""
Utilities used in the different `streamlit` apps.
"""
from typing import Optional

import h5py
import numpy as np
import streamlit as st
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile
from yaml.parser import ParserError

from lyscripts.utils import report_func_state


def safe_yaml_load(file: Optional[UploadedFile] = None) -> dict:
    """Safely load a dictionary from a YAML file."""
    return yaml.safe_load(file) if file is not None else {}


def samples_from_hdf5(file: Optional[UploadedFile] = None) -> np.ndarray:
    """Load a chain of samples from an uploaded HDF5 `file`."""
    if file is not None:
        with h5py.File(file, mode="r") as h5file:
            try:
                samples = h5file["mcmc/chain"][:]
            except KeyError as key_err:
                raise KeyError("Dataset `mcmc` not in the HDF5 file.") from key_err

            new_shape = (samples.shape[0] * samples.shape[1], samples.shape[2])
            flattened_samples = samples.reshape(new_shape)
            return flattened_samples

    return np.array([])



st_load_yaml_params = report_func_state(
    status_msg="Load YAML params...",
    success_msg="Loaded YAML params.",
    status_func=st.spinner,
    success_func=st.success,
    actions={
        ParserError: (False, st.warning, "Invalid YAML file, stopping"),
    }
)(safe_yaml_load)
"""
Function that safely loads YAML parameters and reports the progress along the way.
"""


st_chain_from_hdf5 = report_func_state(
    status_msg="Load HDF5 samples...",
    success_msg="Loaded HDF5 samples.",
    status_func=st.spinner,
    success_func=st.success,
    actions={
        KeyError: (False, st.warning, "Did not find dataset in HDF5 file."),
    }
)(samples_from_hdf5)
"""
Function that safely loads YAML parameters and reports the progress along the way.
"""
