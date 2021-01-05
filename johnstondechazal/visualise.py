#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: johnstondechazal.visualise 

"""
__author__ = 'Ben Johnston'

from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt

from johnstondechazal.history import History


def plot_history(hist: History,
                 ax: matplotlib.axes.Axes,
                 c: Tuple[str, str] = ['b', 'r'],
                 marker='o',
                 s=3):  # pragma: no cover
    """Plot landmark history
    
    :param hist: [description]
    :type hist: History
    :param figure: [description]
    :type figure: plt.figure
    :return: [description]
    :rtype: plt.figure
    """

    num_samples = len(hist)

    # Plot means
    for idx, (mean, *_) in enumerate(hist):

        if (idx + 1) == num_samples:
            # ax.scatter(mean[0], mean[1], s=s, marker='x')
            ax.scatter(mean[0], mean[1], c=c[1], marker=marker)

        elif idx == 0:
            # ax.scatter(mean[0], mean[1], s=s, marker='o')
            ax.scatter(mean[0], mean[1], c=c[0], marker=marker)

        if idx > 0:
            ax.plot([prev_mean[0], mean[0]], [prev_mean[1], mean[1]],
                    c=c[0],
                    linestyle='--')
        prev_mean = mean
