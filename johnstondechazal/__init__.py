"""Top-level package for johnstondechazal."""

__author__ = """Ben Johnston"""
__email__ = 'docEbrown_github@protonmail.com'

from typing import Tuple, Union

import numpy as np

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


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


def select_landmarks(
    precision: np.ndarray,
    landmarks: np.ndarray,
    num_select: Union[int, float] = -1
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Select the landmarks for inclusion in final selection.  The landmarks
    are selected from the annotators with the greatest precision values and the
    selections are made from greatest to least precision.  Annotator's `x` and `y`
    coordinates are considered separately and thus an annotators selection in one
    axis may be selected, but not the corresponding selection in the other axis.
    
    :param precision: The annotator precision metrics for the current mean 
    :type precision: np.ndarray
    :param landmarks: The annotator selected landmarks 
    :type landmarks: np.ndarray
    :param num_select: The number of annotators to select, set to:
        1. A positive integer for a simple selection of the most precise coordinates
        2. A negative integer to select all but the last few annotators
        3. A floating point number to select a percentage of landmarks
        defaults to -1.
    :type num_select: Union[int, float]
    :return: The landmarks and annotators included and removed from the selection 
    :rtype: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    """

    idx_x = precision[:, 0].argsort()[::-1]
    idx_y = precision[:, 1].argsort()[::-1]

    # Apply the selection process
    if isinstance(num_select, float):
        num_select = int(num_select * len(landmarks))

    x_included = idx_x[:num_select]
    y_included = idx_y[:num_select]

    x_excluded = idx_x[num_select:]
    y_excluded = idx_y[num_select:]

    _x = landmarks[x_included, :, 0]
    _y = landmarks[y_included, :, 1]

    _z = np.zeros((len(x_included), landmarks.shape[1], landmarks.shape[2]))

    _z[:, :, 0] = _x
    _z[:, :, 1] = _y

    # Build the included / excluded annator arrays
    included = np.zeros((len(x_included), 2))
    excluded = np.zeros((len(x_excluded), 2))

    included[:, 0] = x_included
    included[:, 1] = y_included

    excluded[:, 0] = x_excluded
    excluded[:, 1] = y_excluded

    return _z, (included, excluded)


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
    prev_mean = np.ones(global_mean.shape) * np.inf

    for idx in range(iterations):

        precision = annotator_precision(landmarks, global_mean)
        weights = precision / precision.sum(axis=0)

        global_mean = np.array([
            np.average(landmarks[:, :, 0].mean(axis=1), weights=weights[:, 0]),
            np.average(landmarks[:, :, 1].mean(axis=1), weights=weights[:, 1]),
        ])

        # Check stop condition
        _stop = abs(global_mean - prev_mean)
        if np.all(_stop < tol):
            break

    return global_mean, precision
