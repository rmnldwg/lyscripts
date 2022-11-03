"""
Testing of the utilities implemented for the plotting routines.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.testing.compare as mpl_comp
import numpy as np
import pytest

from lyscripts.plot.utils import (
    Histogram,
    Posterior,
    _ceil_to_step,
    _floor_to_step,
    draw,
    get_size,
    save_figure,
)


@pytest.fixture
def beta_samples():
    """
    Filename of an HDF5 file where some samples from a Beta distribution are stored
    """
    return "./tests/plot/data/beta_samples.hdf5"


def test_floor_to_step():
    """Check correct rounding down to a given step size."""
    numbers = np.array([0., 3., 7.4, 2.01, np.pi, 12.7, 12.7, 17.3 ])
    steps   = np.array([2 , 2 , 5  , 2   , 3    , 3   , 5   , 0.17 ])
    exp_res = np.array([0., 2., 5. , 2.  , 3.   , 12. , 10. , 17.17])

    comp_res = np.zeros_like(exp_res)
    for i, (num, step) in enumerate(zip(numbers, steps)):
        comp_res[i] = _floor_to_step(num, step)

    assert all(np.isclose(comp_res, exp_res)), "Floor to step did not work properly."


def test_ceil_to_step():
    """Check correct rounding up to a given step size."""
    numbers = np.array([0., 3., 7.4, 2.01, np.pi, 12.7, 12.7, 17.3 ])
    steps   = np.array([2 , 2 , 5  , 2   , 3    , 3   , 5   , 0.17 ])
    exp_res = np.array([2., 4., 10., 4.  , 6.   , 15. , 15. , 17.34])

    comp_res = np.zeros_like(exp_res)
    for i, (num, step) in enumerate(zip(numbers, steps)):
        comp_res[i] = _ceil_to_step(num, step)

    assert all(np.isclose(comp_res, exp_res)), "Ceil to step did not work properly."


def test_histogram_cls(beta_samples):
    """Make sure the histogram data container works as intended."""
    str_filename = beta_samples
    path_filename = Path(str_filename)
    non_existent_filename = "non_existent.hdf5"
    custom_label = "Lorem ipsum"

    hist_from_str = Histogram(filename=str_filename, dataname="beta")
    hist_from_path = Histogram(
        filename=path_filename,
        dataname="beta",
        scale=10.,
        kwargs={"label": custom_label}
    )

    with pytest.raises(FileNotFoundError):
        Histogram(filename=non_existent_filename, dataname="does_not_matter")

    assert np.all(np.isclose(hist_from_str.values, 10. * hist_from_path.values)), (
        "Scaling of data does not work correclty"
    )
    assert np.all(np.isclose(
        hist_from_str.left_percentile(50.),
        hist_from_str.right_percentile(50.),
    )), "50% percentiles should be the same from the left and from the right."
    assert np.all(np.isclose(
        hist_from_path.left_percentile(10.),
        hist_from_path.right_percentile(90.),
    )), "10% from the left is not the same as 90% from the right"
    assert hist_from_str.kwargs["label"] == "beta | mega scan | 100 | ext", (
        "Label extraction did not work"
    )
    assert hist_from_path.kwargs["label"] == custom_label, (
        "Keyword override did not work"
    )


def test_posterior_cls(beta_samples):
    """Test the container class for Beta posteriors."""
    str_filename = beta_samples
    path_filename = Path(str_filename)
    non_existent_filename = "non_existent.hdf5"
    custom_label = "Lorem ipsum"
    x_10 = np.linspace(0., 10., 100)
    x_100 = np.linspace(0., 100., 100)

    post_from_str = Posterior(filename=str_filename, dataname="beta")
    post_from_path = Posterior(
        filename=path_filename,
        dataname="beta",
        scale=10.,
        kwargs={"label": custom_label}
    )

    with pytest.raises(FileNotFoundError):
        Posterior(filename=non_existent_filename, dataname="does_not_matter")

    assert post_from_str.num_success == post_from_path.num_success == 20, (
        "Number of successes not correctly extracted"
    )
    assert post_from_str.num_total == post_from_path.num_total == 40, (
        "Total number of trials not correctly extracted"
    )
    assert post_from_str.num_fail == post_from_path.num_fail == 20, (
        "Number of failures not correctly computed"
    )
    assert np.all(np.isclose(
        10 * post_from_str.pdf(x_100),
        post_from_path.pdf(x_10),
    )), "PDFs with different scaling do not match"
    assert np.all(np.isclose(
        post_from_str.left_percentile(50.),
        post_from_str.right_percentile(50.),
    )), "50% percentiles should be the same from the left and from the right."
    assert np.all(np.isclose(
        post_from_path.left_percentile(10.),
        post_from_path.right_percentile(90.),
    )), "10% from the left is not the same as 90% from the right"


@pytest.mark.mpl_image_compare
def test_draw(beta_samples):
    """Check the drawing function."""
    filename = Path(beta_samples)
    dataname = "beta"
    hist = Histogram(filename, dataname)
    post = Posterior(filename, dataname)
    fig, ax = plt.subplots()
    ax = draw(axes=ax, contents=[hist, post], percent_lims=(2.,2.))
    ax.legend()
    return fig


def test_save_figure(capsys):
    """Check that figures get stored correctly."""
    x = np.linspace(0., 2*np.pi, 200)
    y = np.sin(x)
    fig, ax = plt.subplots(figsize=get_size())
    ax.plot(x,y)
    output_path = "./tests/plot/results/sine"
    formats = ["png", "svg"]
    expected_output = (
        "✓ Saved figure to tests/plot/results/sine in the formats ['png', 'svg'].\n"
    )

    save_figure(fig, output_path, formats)
    save_figure_capture = capsys.readouterr()

    assert mpl_comp.compare_images(
        expected="./tests/plot/baseline/sine.png",
        actual="./tests/plot/results/sine.png",
        tol=0.,
    ) is None, "PNG of figure was not stored correctly."
    assert mpl_comp.compare_images(
        expected="./tests/plot/baseline/sine.svg",
        actual="./tests/plot/results/sine.svg",
        tol=0.,
    ) is None, "SVG of figure was not stored correctly."
    assert save_figure_capture.out == expected_output, (
        "The output during the save figure procedure was wrong."
    )

