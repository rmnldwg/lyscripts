"""Utility functions for the plotting commands."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import field
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpydantic import NDArray, Shape
from pydantic import BaseModel

from lyscripts.decorators import (
    check_input_file_exists,
    check_output_dir_exists,
    log_state,
)

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes as MPLAxes
    from matplotlib.figure import Figure

# define USZ colors
COLORS = {
    "blue": "#005ea8",
    "orange": "#f17900",
    "green": "#00afa5",
    "red": "#ae0060",
    "gray": "#c5d5db",
}
COLOR_CYCLE = cycle(COLORS.values())
CM_PER_INCH = 2.54


def floor_at_decimal(value: float, decimal: int) -> float:
    """Compute the floor of ``value`` for the specified ``decimal``.

    Essentially the distance to the right of the decimal point. May be negative.
    """
    power = 10**decimal
    return np.floor(power * value) / power


def ceil_at_decimal(value: float, decimal: int) -> float:
    """Compute the ceiling of ``value`` for the specified ``decimal``.

    Analog to :py:func:`.floor_at_decimal`, this is the distance to the right of the
    decimal point. May be negative.
    """
    return -floor_at_decimal(-value, decimal)


def floor_to_step(value: float, step: float) -> float:
    """Compute next value on ladder of stepsize ``step`` still below ``value``."""
    return (value // step) * step


def ceil_to_step(value: float, step: float) -> float:
    """Compute next value on ladder of stepsize ``step`` still above ``value``."""
    return floor_to_step(value, step) + step


def clean_and_check(filename: str | Path) -> Path:
    """Check if file with ``filename`` exists.

    If not, raise error, otherwise return cleaned :py:class:`~pathlib.PosixPath`.
    """
    filepath = Path(filename)
    if not filepath.exists():
        msg = f"File with the name {filename} does not exist at {filepath.resolve()}"
        raise FileNotFoundError(msg)
    return filepath


AbstractDistributionT = TypeVar("AbstractDistributionT", bound="AbstractDistribution")


class AbstractDistribution(BaseModel):
    """Abstract class for distributions that should be plotted."""

    scale: float = 100.0
    offset: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=lambda: {})

    @abstractmethod
    def draw(self, axes: MPLAxes) -> MPLAxes:
        """Draw the distribution into the provided ``axes``."""
        ...

    @abstractmethod
    def left_percentile(self, percent: float) -> float:
        """Compute the point where ``percent`` of the values are to the left."""
        ...

    @abstractmethod
    def right_percentile(self, percent: float) -> float:
        """Compute the point where ``percent`` of the values are to the right."""
        ...

    def _get_label(self) -> str:
        """Compute label for when ``kwargs`` does not contain one."""

    @property
    def label(self) -> str:
        """Return the label of the histogram."""
        return self.kwargs.get("label", self._get_label())


class Histogram(AbstractDistribution):
    """Class containing data for plotting a histogram."""

    raw_values: NDArray[Shape["*"], float]  # noqa: F722

    @property
    def values(self) -> np.ndarray:
        """Return the values of the histogram scaled and offset."""
        return self.raw_values * self.scale + self.offset

    @classmethod
    def from_hdf5(
        cls: type[Histogram],
        filename: str | Path,
        dataname: str,
        scale: float = 100.0,
        offset: float = 0.0,
        **kwargs,
    ) -> Histogram:
        """Create a histogram from an HDF5 file."""
        filename = clean_and_check(filename)
        with h5py.File(filename, mode="r") as h5file:
            dataset = h5file[dataname]
            if "label" not in kwargs:
                kwargs["label"] = get_label(dataset.attrs)
            return cls(raw_values=dataset[:], scale=scale, offset=offset, kwargs=kwargs)

    def left_percentile(self, percent: float) -> float:
        """Compute the point where `percent` of the values are to the left."""
        return np.percentile(self.values, percent)

    def right_percentile(self, percent: float) -> float:
        """Compute the point where `percent` of the values are to the right."""
        return np.percentile(self.values, 100.0 - percent)

    def draw(self, axes: MPLAxes, **defaults) -> Any:
        """Draw the histogram into the provided ``axes``."""
        xlim = axes.get_xlim()

        hist_kwargs = defaults.get("hist", {}).copy()
        hist_kwargs.update(self.kwargs)

        if self.label is not None:
            hist_kwargs["label"] = self.label

        return axes.hist(self.values, range=xlim, **hist_kwargs)


class BetaPosterior(AbstractDistribution):
    """Class for storing plot configs for a Beta posterior."""

    num_success: int
    num_total: int

    @classmethod
    def from_hdf5(
        cls: type[BetaPosterior],
        filename: str | Path,
        dataname: str,
        scale: float = 100.0,
        offset: float = 0.0,
        **kwargs,
    ) -> BetaPosterior:
        """Initialize data container for Beta posteriors from HDF5 file."""
        filename = clean_and_check(filename)
        with h5py.File(filename, mode="r") as h5file:
            dataset = h5file[dataname]
            try:
                num_success = int(dataset.attrs["num_match"])
                num_total = int(dataset.attrs["num_total"])
            except KeyError as key_err:
                raise KeyError(
                    "Dataset does not contain observed prevalence data"
                ) from key_err

        return cls(
            num_success=num_success,
            num_total=num_total,
            scale=scale,
            offset=offset,
            kwargs=kwargs,
        )

    def _get_label(self) -> str:
        return f"data: {self.num_success} of {self.num_total}"

    @property
    def num_fail(self):
        """Return the number of failures, i.e. the totals minus the successes."""
        return self.num_total - self.num_success

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the probability density function."""
        return sp.stats.beta.pdf(
            x,
            a=self.num_success + 1,
            b=self.num_fail + 1,
            loc=self.offset,
            scale=self.scale,
        )

    def left_percentile(self, percent: float) -> float:
        """Return the point where the CDF reaches ``percent``."""
        return sp.stats.beta.ppf(
            percent / 100.0,
            a=self.num_success + 1,
            b=self.num_fail + 1,
            scale=self.scale,
        )

    def right_percentile(self, percent: float) -> float:
        """Return the point where 100% minus the CDF equals ``percent``."""
        return sp.stats.beta.ppf(
            1.0 - (percent / 100.0),
            a=self.num_success + 1,
            b=self.num_fail + 1,
            scale=self.scale,
        )

    def draw(self, axes: MPLAxes, resolution: int = 300, **defaults) -> Any:
        """Draw the Beta posterior into the provided ``axes``.

        Returns a handle and a label for the legend.
        """
        left, right = axes.get_xlim()
        x = np.linspace(left, right, resolution)
        y = self.pdf(x)

        plot_kwargs = defaults.get("plot", {}).copy()
        plot_kwargs.update(self.kwargs)

        if self.label is not None:
            plot_kwargs["label"] = self.label

        return axes.plot(x, y, **plot_kwargs)


def get_size(width="single", unit="cm", ratio="golden"):
    """Return a tuple of figure sizes in inches.

    This is provided as the ``matplotlib`` keyword argument ``figsize`` expects it.
    This figure size is computed from a ``width``, in the ``unit`` of centimeters by
    default, and a ``ratio`` which is set to the golden ratio by default.

    >>> get_size(width="single", ratio="golden")
    (3.937007874015748, 2.4332557935820445)
    >>> get_size(width="full", ratio=2.)
    (6.299212598425196, 3.149606299212598)
    >>> get_size(width=10., ratio=1.)
    (3.937007874015748, 3.937007874015748)
    >>> get_size(width=5, unit="inches", ratio=2./3.)
    (5, 7.5)
    """
    if width == "single":
        width = 10
    elif width == "full":
        width = 16

    ratio = 1.618 if ratio == "golden" else ratio
    width = width / CM_PER_INCH if unit == "cm" else width
    height = width / ratio
    return (width, height)


def get_label(attrs: Mapping) -> str:
    """Extract label of a histogram from the HDF5 ``attrs`` object of the dataset."""
    label = []
    transforms = {
        "label": str,
        "modality": str,
        "t_stage": str,
        "midline_ext": lambda x: "ext" if x else "noext",
    }
    for key, func in transforms.items():
        if key in attrs and attrs[key] is not None:
            label.append(func(attrs[key]))
    return " | ".join(label)


def get_xlims(
    contents: AbstractDistributionT,
    percent_lims: tuple[float] = (10.0, 10.0),
) -> tuple[float]:
    """Get the x-axis limits for a plot containing multiple distribution.

    Compute the ``xlims`` of a plot containing histograms and probability density
    functions by considering their smallest and largest percentiles.
    """
    left_percentiles = np.array(
        [c.left_percentile(percent_lims[0]) for c in contents],
    )
    left_lim = np.min(left_percentiles)
    right_percentiles = np.array(
        [c.right_percentile(percent_lims[0]) for c in contents],
    )
    right_lim = np.max(right_percentiles)
    return left_lim, right_lim


def draw(
    axes: MPLAxes,
    contents: list[AbstractDistribution],
    percent_lims: tuple[float, float] = (10.0, 10.0),
    xlims: tuple[float] | None = None,
    hist_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> MPLAxes:
    """Draw histograms and Beta posterior from ``contents`` into ``axes``.

    The limits of the x-axis is computed to be the smallest and largest left and right
    percentile of all provided ``contents`` respectively via the ``percent_lims`` tuple.

    The ``hist_kwargs`` define general settings that will be applied to all histograms.
    One additional key ``'nbins'`` may be used to adjust only the numbers, not the
    spacing of the histogram bins.
    Similarly, ``plot_kwargs`` adjusts the default settings for the Beta posteriors.

    Both these keyword arguments can be overwritten by what the individual ``contents``
    have defined.
    """
    if not all(isinstance(c, AbstractDistribution) for c in contents):
        raise TypeError("Contents must be subclasses of `AbstractDistribution`")

    xlims = xlims or get_xlims(contents, percent_lims)

    if len(xlims) != 2 or xlims[0] > xlims[-1]:
        raise ValueError("`xlims` must be tuple of two increasing values")

    axes.set_xlim(*xlims)

    default_kwargs = {
        "hist": {
            "density": True,
            "histtype": "stepfilled",
            "alpha": 0.7,
            "bins": 50,
        },
        "plot": {},
    }
    default_kwargs["hist"].update(hist_kwargs or {})
    default_kwargs["plot"].update(plot_kwargs or {})

    for content in contents:
        content.draw(axes, **default_kwargs)

    return axes


def split_legends(
    axes: MPLAxes,
    titles: list[str],
    locs: list[tuple[float, float]],
    **kwargs,
) -> None:
    """Separate labels in ``axes`` into separate legends with ``titles`` at ``locs``."""
    legend_kwargs = {
        "title_fontsize": "small",
        "labelspacing": 0.1,
        "loc": "upper left",
    }
    legend_kwargs.update(kwargs)

    handles, labels = axes.get_legend_handles_labels()
    labels_per_legend = len(labels) // len(titles)

    for i, (title, loc) in enumerate(zip(titles, locs, strict=True)):
        start = i * labels_per_legend
        stop = (i + 1) * labels_per_legend if i < len(titles) - 1 else None
        idx = slice(start, stop)

        legend = axes.legend(
            handles[idx],
            labels[idx],
            bbox_to_anchor=loc,
            title=title,
            **legend_kwargs,
        )
        axes.add_artist(legend)


@log_state()
@check_input_file_exists
def use_mpl_stylesheet(file_path: str | Path):
    """Load a ``.mplstyle`` stylesheet from ``file_path``."""
    plt.style.use(file_path)


@log_state()
@check_output_dir_exists
def save_figure(
    output_path: str | Path,
    figure: Figure,
    formats: list[str] | None,
):
    """Save a ``figure`` to ``output_path`` in every one of the provided ``formats``."""
    for frmt in formats:
        figure.savefig(output_path.with_suffix(f".{frmt}"))
