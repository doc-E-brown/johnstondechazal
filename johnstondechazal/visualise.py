#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: johnstondechazal.visualise

"""
__author__ = 'Ben Johnston'

from typing import Tuple

import matplotlib.axes

from johnstondechazal.history import History


def plot_history(hist: History,
                 ax: matplotlib.axes.Axes,
                 c: Tuple[str, str] = ['b', 'r'],
                 marker='o'):  # pragma: no cover
    """Plot landmark history

    :param hist: [description]
    :type hist: History
    :param figure: [description]
    :type figure: plt.figure
    :return: [description]
    :rtype: plt.figure
    """

    num_samples = len(hist)
    points = []

    # Plot means
    for idx, (mean, *_) in enumerate(hist):

        if (idx + 1) == num_samples:
            points.append(
                ax.scatter(mean[0],
                           mean[1],
                           c=c[1],
                           marker=marker,
                           label='Final Location'))

        elif idx == 0:
            points.append(
                ax.scatter(mean[0],
                           mean[1],
                           c=c[0],
                           marker=marker,
                           label='Global Mean'))

        if idx > 0:
            points.append(
                ax.plot([prev_mean[0], mean[0]], [prev_mean[1], mean[1]],
                        c=c[0],
                        linestyle='--'))
        prev_mean = mean

    return points
