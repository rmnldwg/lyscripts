"""
Visualize the component assignments of the trained mixture model.
"""


import argparse
import logging
import numpy as np
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from lyscripts.plot.utils import COLORS, SUBSITE_COLORS, save_figure, add_perpendicular_ticks, p_to_xyz, add_perpendicular_crosses_3d
from lyscripts.utils import load_yaml_params
from matplotlib.ticker import StrMethodFormatter

logger = logging.getLogger(__name__)

def _add_parser(
    subparsers: argparse._SubParsersAction,
    help_formatter,
):
    """Add an ``ArgumentParser`` to the subparsers action."""
    parser = subparsers.add_parser(
        Path(__file__).name.replace(".py", ""),
        description=__doc__,
        help=__doc__,
        formatter_class=help_formatter,
    )
    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser):
    """Add arguments to the parser."""
    parser.add_argument(
        "--input", type=Path,
        help="File path of the mixture coefficients"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path for the plot"
    )
    parser.add_argument(
        "-p", "--params", default="./params.yaml", type=Path,
        help="Path to parameter file."
    )
    parser.set_defaults(run_main=main)


# def plot_2d_simplex(mixture_df, data):
#     _, bottom_ax = plt.subplots()
#     subsites = list(mixture_df.columns)
#     cluster_x = mixture_df.loc[0]
#     cluster_y = [0. for _ in subsites]
#     annotations = [f"{label}\n({num})" for label, num in num_patients.items()]
#     bottom_ax.scatter(
#         cluster_x, cluster_y,
#         s=[num for num in num_patients.values()],
#         c=list(generate_location_colors(subsites)),
#         alpha=0.7,
#         linewidths=0.,
#         zorder=10,
#     )

#     sorted_idx = cluster_components.argsort()
#     sorted_x = cluster_components[sorted_idx]
#     sorted_annotations = [annotations[i] for i in sorted_idx]
#     sorted_num = [list(num_patients.values())[i] for i in sorted_idx]
#     for i, (x, num, annotation) in enumerate(zip(sorted_x, sorted_num, sorted_annotations)):
#         bottom_ax.annotate(
#             annotation,
#             # sqrt, because marker's area grows linearly with patient num, not radius
#             xy=(x, np.sqrt(0.0000003 * num) * (- 1)**i),
#             xytext=(x, 0.025 * (- 1)**i),
#             ha="center",
#             va="bottom" if i % 2 == 0 else "top",
#             fontsize="small",
#             arrowprops={
#                 "arrowstyle": "-",
#                 "color": USZ["gray"],
#                 "linewidth": 1.,
#             }
#         )

#     bottom_ax.set_xlabel("assignment to component A")
#     bottom_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
#     top_ax = bottom_ax.secondary_xaxis(
#         location="top",
#         functions=(lambda x: 1. - x, lambda x: 1. - x),
#     )
#     top_ax.set_xlabel("assignment to component B")
#     top_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
#     bottom_ax.set_yticks([])
#     bottom_ax.grid(axis="x", alpha=0.5, color=USZ["gray"], linestyle=":")
#     plt.savefig(args.output, bbox_inches="tight", dpi=300)

def plot_3d_simplex(mixture_df, data, output, component_names = False):
    data = pd.read_csv(data, header=[0, 1, 2])
    subsites = list(mixture_df.columns)
    colors_ordered = [SUBSITE_COLORS[subsite] for subsite in subsites]

    # Define the plane's normal vector
    normal_vector = np.array([1,1,1])/np.sqrt(3)

    v1 = np.array([1,-1,0])/np.sqrt(2)

    # Calculate the second orthogonal vector using the cross product
    v2 = np.cross(normal_vector, v1) *-1

    # Project the point onto the new coordinate system
    origin = np.array([0, 1, 0])
    x_origin =  origin @ v1
    y_origin = origin @ v2

    x_vals = mixture_df.T @ v1 - x_origin
    y_vals = mixture_df.T @ v2 - y_origin

    extremes = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
    extremes_x = extremes @ v1 - x_origin
    extremes_y = extremes @ v2 - y_origin

    # Plot the point in 2D
    import matplotlib.pyplot as plt

    odered_value_counts = data['tumor']['1']['subsite'].value_counts()[subsites]
    sizes = odered_value_counts * 3

    fig, ax = plt.subplots(figsize=(8, 6.8))

    # Plot the points with varying sizes and colors
    for i in range(len(x_vals)):
        ax.scatter(x_vals[i], y_vals[i], s=sizes[i], color=colors_ordered[i], label=subsites[i])
        ax.text(x_vals[i], y_vals[i], subsites[i], fontsize=9, ha='center', va='center')

    legend_text = []
    for index in range(len(subsites)):
        legend_text.append(subsites[index] + ', ' + str(odered_value_counts[index]) + ' patients')

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subsite)
                    for color, subsite in zip(colors_ordered, legend_text)]

    # Add a legend with fixed dot sizes
    ax.legend(handles=legend_elements, loc='upper right', title='Subsites', fontsize='small')

    # Connect the points
    ax.plot(extremes_x, extremes_y, color='black', alpha=0.5)

    # Close the triangle by connecting the last point to the first
    ax.plot([extremes_x[-1], extremes_x[0]], [extremes_y[-1], extremes_y[0]], color='black', alpha=0.5)

    # Calculate midpoints of each side of the triangle
    midpoints_x = (extremes_x[0] + extremes_x[1]) / 2, (extremes_x[1] + extremes_x[2]) / 2, (extremes_x[2] + extremes_x[0]) / 2
    midpoints_y = (extremes_y[0] + extremes_y[1]) / 2, (extremes_y[1] + extremes_y[2]) / 2, (extremes_y[2] + extremes_y[0]) / 2

    # Draw lines from each vertex to the midpoint of the opposite side
    ax.plot([extremes_x[0], midpoints_x[1]], [extremes_y[0], midpoints_y[1]], color='gray', linestyle='--', linewidth=1)
    ax.plot([extremes_x[1], midpoints_x[2]], [extremes_y[1], midpoints_y[2]], color='gray', linestyle='--', linewidth=1)
    ax.plot([extremes_x[2], midpoints_x[0]], [extremes_y[2], midpoints_y[0]], color='gray', linestyle='--', linewidth=1)

    # Add perpendicular ticks to each line with adjusted length
    add_perpendicular_ticks(extremes_x[0], extremes_y[0], midpoints_x[1], midpoints_y[1])
    add_perpendicular_ticks(extremes_x[1], extremes_y[1], midpoints_x[2], midpoints_y[2])
    add_perpendicular_ticks(extremes_x[2], extremes_y[2], midpoints_x[0], midpoints_y[0])

    # Scaling factor to move the text farther from the vertices
    scaling_factor = 1.1

    # Calculate the centroid of the triangle
    centroid_x = np.mean(extremes_x)
    centroid_y = np.mean(extremes_y)

    # Scale the label positions away from the centroid
    scaled_extremes_x = centroid_x + scaling_factor * (extremes_x - centroid_x)
    scaled_extremes_y = centroid_y + scaling_factor * (extremes_y - centroid_y)

    if component_names:
        # Plot the text labels farther away from the triangle
        if 'C32.0' in subsites:
            component_larynx = mixture_df['C32.0'].argmax()
            ax.text(scaled_extremes_x[component_larynx], scaled_extremes_y[component_larynx], "Larynx like", 
                    fontsize=10, ha='right', va='top', c=COLORS['orange'])

        if 'C01' in subsites:
            component_oropharynx = mixture_df['C01'].argmax()
            ax.text(scaled_extremes_x[component_oropharynx], scaled_extremes_y[component_oropharynx], "Oropharynx like", 
                    fontsize=10, ha='left', va='top', c=COLORS['green'])

        if 'C03' in subsites:
            component_oral_cavity = mixture_df['C03'].argmax()
            ax.text(scaled_extremes_x[component_oral_cavity], scaled_extremes_y[component_oral_cavity], "Oral cavity like", 
                    fontsize=10, ha='left', va='top', c=COLORS['blue'])

        if 'C13' in subsites:
            component_hypopharynx = mixture_df['C13'].argmax()
            ax.text(scaled_extremes_x[component_hypopharynx], scaled_extremes_y[component_hypopharynx], "Hypopharynx like", 
                    fontsize=10, ha='left', va='top', c=COLORS['red'])

    save_figure(output, fig, formats=["png", "svg"])
    logger.info(f"Simplex plot saved")

def main(args: argparse.Namespace):
    mixture_df = pd.read_csv(args.input)
    nr_components = len(mixture_df)
    params = load_yaml_params(args.params)
    data = params['general']['data']
    if nr_components == 2:
        plot_2d_simplex(mixture_df, data)
    elif nr_components == 3:
        plot_3d_simplex(mixture_df, data, output = args.output)
    else:
        logger.info(f"Simplex not supported for {nr_components} components")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)

    args = parser.parse_args()
    args.run_main(args)

