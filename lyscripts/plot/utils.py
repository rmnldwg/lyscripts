"""Utility functions for the plotting commands."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.colors import LinearSegmentedColormap

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
    "light_blue": "#00A8D8",
    "dark_grey_experimental": "#404756",
}
COLOR_CYCLE = cycle(COLORS.values())
CM_PER_INCH = 2.54
blue_to_white = LinearSegmentedColormap.from_list(
    "blue to white", [COLORS["blue"], "#ffffff"], N=256
)
green_to_white = LinearSegmentedColormap.from_list(
    "green_to_white", [COLORS["green"], "#ffffff"], N=256
)
red_to_white = LinearSegmentedColormap.from_list(
    "red_to_white", [COLORS["red"], "#ffffff"], N=256
)
orange_to_white = LinearSegmentedColormap.from_list(
    "orange_to_white", [COLORS["orange"], "#ffffff"], N=256
)
SUBSITE_COLORS = {
    "C03": blue_to_white(0),
    "C04": blue_to_white(0.15),
    "C06": blue_to_white(0.3),
    "C02": blue_to_white(0.45),
    "C05": blue_to_white(0.6),
    "C10": green_to_white(0),
    "C09": green_to_white(0.3),
    "C01": green_to_white(0.6),
    "C12": red_to_white(0),
    "C13": red_to_white(0.5),
    "C32.1": orange_to_white(0),
    "C32.2": orange_to_white(0.3),
    "C32.0": orange_to_white(0.6),
}


def floor_at_decimal(value: float, decimal: int) -> float:
    """Compute the floor of ``value`` for the specified ``decimal``.

    Essentially, this is the distance to the right of the decimal point. May be negative.
    """
    power = 10**decimal
    return np.floor(power * value) / power


def ceil_at_decimal(value: float, decimal: int) -> float:
    """Compute the ceiling of ``value`` for the specified ``decimal``.

    Analog to :py:func:`._floot_at_decimal`, this is the distance to the right of the
    decimal point. May be negative.
    """
    return -floor_at_decimal(-value, decimal)


def floor_to_step(value: float, step: float) -> float:
    """Compute next closest value on ladder of stepsize ``step`` that is below ``value``."""
    return (value // step) * step


def ceil_to_step(value: float, step: float) -> float:
    """Compute next closest value on ladder of stepsize ``step`` that is above ``value``."""
    return floor_to_step(value, step) + step


def clean_and_check(filename: str | Path) -> Path:
    """Check if file with ``filename`` exists.

    If not, raise error, otherwise return cleaned ``PosixPath``.
    """
    filepath = Path(filename)
    if not filepath.exists():
        msg = f"File with the name {filename} does not exist at {filepath.resolve()}"
        raise FileNotFoundError(msg)
    return filepath


AbstractDistributionT = TypeVar("AbstractDistributionT", bound="AbstractDistribution")


@dataclass(kw_only=True)
class AbstractDistribution:
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


@dataclass(kw_only=True)
class Histogram(AbstractDistribution):
    """Class containing data for plotting a histogram."""

    values: np.ndarray

    def __post_init__(self) -> None:  # noqa: D105
        self.values = self.scale * self.values + self.offset

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
            return cls(values=dataset[:], scale=scale, offset=offset, kwargs=kwargs)

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


@dataclass(kw_only=True)
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


def p_to_xyz(p):
    # Project 4D representation down to 3D representation
    s3 = 1 / math.sqrt(3.0)
    s6 = 1 / math.sqrt(6.0)
    x = -1 * p[0] + 1 * p[1] + 0 * p[2] + 0 * p[3]
    y = -s3 * p[0] - s3 * p[1] + 2 * s3 * p[2] + 0 * p[3]
    z = -s3 * p[0] - s3 * p[1] - s3 * p[2] + 3 * s6 * p[3]
    return x, y, z


def add_perpendicular_crosses_3d(ax, x1, y1, z1, x2, y2, z2, tick_length=0.03):
    num_ticks = 6  # Number of ticks
    for i in range(num_ticks):
        t = i / (num_ticks - 1)

        # Interpolation point (x, y, z) on the line
        x_tick = x1 + t * (x2 - x1)
        y_tick = y1 + t * (y2 - y1)
        z_tick = z1 + t * (z2 - z1)

        # Vector along the line (direction of the line)
        line_vec = np.array([x2 - x1, y2 - y1, z2 - z1])

        # First perpendicular vector (cross product with z-axis)
        perp_vec1 = np.cross(line_vec, [0, 0, 1])
        length1 = np.linalg.norm(perp_vec1)
        if (
            length1 == 0
        ):  # Prevent division by zero (in rare cases when parallel to z-axis)
            perp_vec1 = np.cross(line_vec, [1, 0, 0])  # Cross with x-axis instead
            length1 = np.linalg.norm(perp_vec1)
        perp_vec1 /= length1  # Normalize

        # Second perpendicular vector (cross product with line_vec and perp_vec1)
        perp_vec2 = np.cross(line_vec, perp_vec1)
        perp_vec2 /= np.linalg.norm(perp_vec2)  # Normalize

        # Scale the perpendicular vectors by tick length
        perp_vec1 *= tick_length
        perp_vec2 *= tick_length

        # Draw the cross (two perpendicular lines)
        ax.plot(
            [x_tick - perp_vec1[0], x_tick + perp_vec1[0]],
            [y_tick - perp_vec1[1], y_tick + perp_vec1[1]],
            [z_tick - perp_vec1[2], z_tick + perp_vec1[2]],
            color="gray",
            linewidth=0.8,
        )

        ax.plot(
            [x_tick - perp_vec2[0], x_tick + perp_vec2[0]],
            [y_tick - perp_vec2[1], y_tick + perp_vec2[1]],
            [z_tick - perp_vec2[2], z_tick + perp_vec2[2]],
            color="gray",
            linewidth=0.8,
        )

        ax.text(
            x_tick,
            y_tick,
            z_tick,
            f"{int(100 - t * 100)}%",
            fontsize=6,
            ha="right",
            va="bottom",
        )


def add_perpendicular_ticks(x1, y1, x2, y2, tick_length=0.01):
    num_ticks = 6  # Number of ticks including 0% and 100%
    for i in range(num_ticks):
        t = i / (num_ticks - 1)
        x_tick = x1 + t * (x2 - x1)
        y_tick = y1 + t * (y2 - y1)

        # Vector along the line
        dx = x2 - x1
        dy = y2 - y1

        # Perpendicular vector
        perp_dx = -dy
        perp_dy = dx

        # Normalize the perpendicular vector
        length = np.sqrt(perp_dx**2 + perp_dy**2)
        perp_dx /= length
        perp_dy /= length

        # Draw tick as a short perpendicular line
        plt.plot(
            [x_tick - tick_length * perp_dx, x_tick + tick_length * perp_dx],
            [y_tick - tick_length * perp_dy, y_tick + tick_length * perp_dy],
            color="gray",
            linewidth=0.8,
        )
        plt.text(
            x_tick,
            y_tick,
            f"{int(100 - t * 100)}%",
            fontsize=8,
            ha="right",
            va="bottom",
        )
