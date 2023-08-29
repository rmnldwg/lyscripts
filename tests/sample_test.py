"""
Test the sampling command with some example patients.

Originally, I wanted to test that the sampling procedure is reproducible, but the
`emcee` package does not seem to work with any kind of seed in a reproducible manner.

Maybe I am doing something wrong...
"""
import numpy as np
import pandas as pd
import pytest
from emcee.backends import Backend, HDFBackend
from lymph import Unilateral

from lyscripts.sample import run_mcmc_with_burnin
from lyscripts.utils import (
    LymphModel,
    create_model_from_config,
    load_data_for_model,
    load_yaml_params,
)


@pytest.fixture
def params() -> dict:
    """Fixture providing stored `test_params.yaml` file."""
    return load_yaml_params("./tests/test_params.yaml")


@pytest.fixture
def model(params: dict) -> LymphModel:
    """Model fixture from parameters."""
    return create_model_from_config(params)


@pytest.fixture
def data(model: LymphModel) -> pd.DataFrame:
    """Get synthetically generated data from disk."""
    header_rows = [0,1] if isinstance(model, Unilateral) else [0,1,2]
    return load_data_for_model("./tests/test_data.csv", header_rows=header_rows)


@pytest.fixture
def backend() -> Backend:
    """Provide a non-persistent backend to store samples during the test run."""
    return Backend()


@pytest.fixture
def stored_hdf5_backend() -> HDFBackend:
    """Load previously sampled backend for comparison."""
    return HDFBackend("./tests/test_backend.hdf5", read_only=True)


def test_sampling(
    model: LymphModel,
    data: pd.DataFrame,
    params: dict,
    backend: Backend,
    stored_hdf5_backend: HDFBackend,
):
    """Test the basic sampling function."""
    model.patient_data = data
    ndim = len(model.spread_probs) + model.diag_time_dists.num_parametric
    nwalker = ndim * params["sampling"]["walkers_per_dim"]

    def log_prob_fn(theta: np.ndarray) -> float:
        """The log-probability function to sample from."""
        return model.likelihood(given_params=theta)

    np.random.seed(128)
    info = run_mcmc_with_burnin(
        nwalker, ndim, log_prob_fn,
        persistent_backend=backend,
        nsteps=params["sampling"]["nsteps"],
        burnin=params["sampling"]["burnin"],
        keep_burnin=False,
        npools=0,
    )

    actual_chain = backend.get_chain(flat=True)
    expected_chain = stored_hdf5_backend.get_chain(flat=True)
    assert np.all(np.isclose(actual_chain, expected_chain)), "Chain was not reproduced"
