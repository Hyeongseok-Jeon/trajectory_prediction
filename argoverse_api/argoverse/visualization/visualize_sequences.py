#!/usr/bin/env python
import os
import sys

cur_path = os.path.abspath(__file__)
root = os.path.dirname(os.path.dirname(os.path.dirname(cur_path)))
sys.path.append(root)

"""A simple python script template."""
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp

from argoverse.map_representation.map_api import ArgoverseMap

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}


def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def viz_sequence(
        df: pd.DataFrame,
        lane_centerlines: Optional[List[np.ndarray]] = None,
        show: bool = True,
        axes=None,
        smoothen: bool = False,
) -> None:
    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    if axes == None:
        plt.figure(0, figsize=(8, 7))

    x_min = min(df["X"][df['OBJECT_TYPE'] == 'AGENT'])-20
    x_max = max(df["X"][df['OBJECT_TYPE'] == 'AGENT'])+20
    y_min = min(df["Y"][df['OBJECT_TYPE'] == 'AGENT'])-20
    y_max = max(df["Y"][df['OBJECT_TYPE'] == 'AGENT'])+20

    if lane_centerlines is None:
        if axes == None:
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
        else:
            axes.set_xlim(x_min, x_max)
            axes.set_ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                    np.min(lane_cl[:, 0]) < x_max
                    and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        if axes == None:
            plt.plot(
                lane_cl[:, 0],
                lane_cl[:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                zorder=0,
            )
        else:
            axes.plot(
                lane_cl[:, 0],
                lane_cl[:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                zorder=0,
            )
    frames = df.groupby("TRACK_ID")

    if axes == None:
        plt.xlabel("Map X")
        plt.ylabel("Map Y")
    else:
        axes.set_xlabel("Map X")
        axes.set_ylabel("Map Y")

    color_dict = {"AGENT": "#0000ff", "OTHERS": "#d3e8ef", "AV": "#ff0000"}
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]
        if object_type == "AV":
            object_type = "OTHERS"

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]
        if axes == None:
            plt.plot(
                cor_x,
                cor_y,
                "-",
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                linewidth=1,
                zorder=_ZORDER[object_type],
            )
        else:
            if object_type == "AGENT":
                axes.plot(
                    cor_x,
                    cor_y,
                    linestyle = (0, (3, 3)),
                    color=color_dict[object_type],
                    label=object_type if not object_type_tracker[object_type] else "",
                    alpha=1,
                    linewidth=2,
                    zorder=_ZORDER[object_type],
                )
            else:
                axes.plot(
                    cor_x,
                    cor_y,
                    "-",
                    color=color_dict[object_type],
                    label=object_type if not object_type_tracker[object_type] else "",
                    alpha=1,
                    linewidth=1,
                    zorder=_ZORDER[object_type],
                )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 7
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7
        if axes == None:
            plt.plot(
                final_x,
                final_y,
                marker_type,
                color=color_dict[object_type],
                label=object_type if not object_type_tracker[object_type] else "",
                alpha=1,
                markersize=marker_size,
                zorder=_ZORDER[object_type],
            )
        else:
            if object_type == "AGENT":
                pass
            else:
                axes.plot(
                    final_x,
                    final_y,
                    marker_type,
                    color=color_dict[object_type],
                    label=object_type if not object_type_tracker[object_type] else "",
                    alpha=1,
                    markersize=marker_size,
                    zorder=_ZORDER[object_type],
                )

        object_type_tracker[object_type] += 1
    if axes == None:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    else:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=7,
        label="Others",
    )
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")
    if axes == None:
        # plt.axis("off")
        if show:
            plt.show()
    # else:
    #     axes.axis("off")
