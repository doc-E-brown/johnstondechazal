#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule::johnstondechazal.method

"""

__author__ = 'Ben Johnston'

from typing import Callable, Tuple

import numpy as np


def annotator_precision(vals: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Compute annotator precision

    :param vals: Annotator selected landmarks
    :type vals: np.ndarray
    :param mean: The current landmark mean
    :type mean: np.ndarray
    :return: The precision of the annotators x, y coordinate selections
    :rtype: np.ndarray
    """

    update = np.sqrt((vals - mean)**2) + np.finfo(float).eps
    return 1 / update.mean(axis=1)


def find_worst_sum(precision: np.ndarray) -> Tuple[Tuple[int], Tuple[int]]:
    """Drop the worst performing annotator by summing the precision values for the
    both the x and y directions and eliminating the annotator with the lowest
    precision score

    :param precision: The annotator precision values
    :type precision: np.ndarray
    :return: The annotators included and the annotator to exclude from the
        sample
    :rtype: Tuple[Tuple[int], Tuple[int]]
    """

    precision_sum = precision.sum(axis=1)
    indices = precision_sum.argsort().tolist()
    worst_annot = indices.pop(0)

    return tuple([tuple(indices), tuple([worst_annot])])


def select_landmarks(
    precision: np.ndarray,
    landmarks: np.ndarray,
    select_func: Callable = find_worst_sum,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Select the landmarks for inclusion in final selection.  The landmarks
    are selected from the annotators with the greatest precision values and the
    selections are made from greatest to least precision.  Annotator's `x` and
    `y` coordinates are considered separately and thus an annotators selection
    in one axis may be selected, but not the corresponding selection in the
    other axis.

    :param precision: The annotator precision metrics for the current mean
    :type precision: np.ndarray
    :param landmarks: The annotator selected landmarks
    :type landmarks: np.ndarray
    :param select_func: The function to call to determine which annotators to
        remove and which to exclude.  Must have a function signature of:
        `def func(precision: np.ndarray) -> Tuple[Tuple[int], Tuple[int]]`
        where the first element of the resulting tuple contains the indices of
        the included annotators and the second tuple the excluded annotators
    :return: The landmarks and annotators included and removed from the
        selection
    :rtype: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    """

    idx_include, idx_exclude = select_func(precision)
    new_landmarks = landmarks[idx_include, ]

    return new_landmarks, (idx_include, idx_exclude)


def converge_mean(landmarks: np.ndarray,
                  iterations: int = 20,
                  tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Converge upon an estimate of the global mean, computed as a weighted mean of
    annotator precision.

    :param landmarks: The annotator selected landmarks
    :type landmarks: np.ndarray
    :param iterations: Number iterations to execute, defaults to 20
    :type iterations: int, optional
    :param tol: If changes in mean position are less than the specified
        value convergence terminates, defaults to 1e-4
    :type tol: float, optional
    :return: The converged global mean and the corresponding annotator
        precision values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    global_mean = landmarks.mean(axis=0).mean(axis=0)

    # Fake the size of the previous mean for the first iteration
    prev_mean = np.inf * global_mean

    for idx in range(iterations):

        precision = annotator_precision(landmarks, global_mean)
        weights = precision / precision.sum(axis=0)

        global_mean = np.array([
            np.average(landmarks[:, :, 0].mean(axis=1), weights=weights[:, 0]),
            np.average(landmarks[:, :, 1].mean(axis=1), weights=weights[:, 1]),
        ])

        # # Check stop condition
        stop = np.abs(global_mean - prev_mean)
        if np.any(stop < tol):
            return global_mean, precision

        prev_mean = np.copy(global_mean)

    return global_mean, precision
