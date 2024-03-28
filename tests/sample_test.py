"""
Test the sampling command with some example patients.

Originally, I wanted to test that the sampling procedure is reproducible, but the
`emcee` package does not seem to work with any kind of seed in a reproducible manner.

Maybe I am doing something wrong...
"""
# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from emcee import EnsembleSampler, backends
from lymph import types

from lyscripts.sample import get_starting_state, run_burnin
from lyscripts.utils import create_model, load_patient_data, load_yaml_params


@pytest.fixture
def params() -> dict:
    """Fixture providing stored `test_sample_params.yaml` file."""
    return load_yaml_params("./tests/test_sample_params.yaml")


@pytest.fixture
def model(params: dict) -> types.Model:
    """Model fixture from parameters."""
    return create_model(params)


@pytest.fixture
def loaded_model(model: types.Model) -> types.Model:
    """Get synthetically generated data from disk."""
    data = load_patient_data("./tests/test_data.csv")
    model.load_patient_data(data)
    return model


@pytest.fixture
def hdf5_backend(tmp_path) -> backends.HDFBackend:
    """Load previously sampled backend for comparison."""
    return backends.HDFBackend(tmp_path / "tmp.hdf5")


@pytest.fixture
def sampler(model: types.Model, hdf5_backend: backends.HDFBackend) -> EnsembleSampler:
    """Construct a sampler for the model."""
    np.random.seed(42)
    ndim = model.get_num_dims()
    nwalkers = ndim * 10
    return EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=model.likelihood,
        backend=hdf5_backend,
    )


def test_burnin(sampler: EnsembleSampler):
    """Test the burnin function."""
    burnin_history = run_burnin(
        sampler=sampler,
        burnin=100,
        check_interval=10,
    )
    assert sampler.iteration == 100, "Burnin di not run 100 iterations."
    assert len(burnin_history.steps) == 10, "Burnin history does not have 10 entries."
    assert np.all(
        np.array([
            0.7147557514447068, 0.9227188150264771,
            0.2629624184410706, 0.6001184115584288,
        ])
        == sampler.get_last_sample().coords[0]
    ), "Not reproducible."


def test_get_starting_state(sampler: EnsembleSampler):
    """Test if the starting state can be retrieved."""
    state = get_starting_state(sampler)
    with pytest.raises(AttributeError):
        sampler.get_last_sample()
    assert state.shape == (sampler.nwalkers, sampler.ndim), "State has wrong shape."

    _ = run_burnin(
        sampler=sampler,
        burnin=10,
        check_interval=2,
    )
    assert np.all(
        get_starting_state(sampler).coords
        == sampler.get_last_sample().coords
    ), "State is not the same."
