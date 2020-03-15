#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method tests

"""
__author__ = 'Ben Johnston'

import numpy as np
import pandas as pd
import pytest

from johnstondechazal import annotator_precision, select_landmarks


@pytest.fixture
def estimated_landmarks():
    annotator_estimates = np.ones((3, 2, 2))
    annotator_estimates[0, :, :] = np.array([-1, -1])
    annotator_estimates[1, :, :] = np.array([1, 3])
    annotator_estimates[2:, :] = np.array([2.1, 5])
    return annotator_estimates


def test_annotator_precision(estimated_landmarks):
    """Test compute annotator precision"""

    mean = np.array([[2, 4]])

    expected_precision = np.array([[0.33333, 0.2], [1, 1], [10, 1]])

    precision = annotator_precision(estimated_landmarks, mean)

    np.testing.assert_almost_equal(precision, expected_precision, decimal=2)


def test_for_div_zero(estimated_landmarks):
    """Test for divide by zero error"""

    mean = np.array([[2, 4]])

    precision = annotator_precision(estimated_landmarks, mean)

    assert not np.any(np.isinf(precision))


def test_select_landmarks():
    """Test selecting annotators from precision"""

    precision = np.array([[0.33333, 0.2], [1, 1.2], [10, 1.3], [5, 6]])
    landmarks = np.ones((4, 2, 2))

    landmarks, (included, excluded) = select_landmarks(precision, landmarks)

    assert landmarks.shape == (3, 2, 2)

    expected_included = np.array([[2, 3], [3, 2], [1, 1]])

    np.testing.assert_equal(included, expected_included)

    expected_excluded = np.array([[0, 0]])

    np.testing.assert_equal(excluded, expected_excluded)
