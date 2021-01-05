#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method tests

"""
__author__ = 'Ben Johnston'

from unittest.mock import patch

import numpy as np
import pytest

from johnstondechazal.method import (annotator_precision, converge_mean,
                                     select_landmarks)


@pytest.fixture
def estimated_landmarks():
    annotator_estimates = np.ones((3, 2, 2))
    annotator_estimates[0, :, :] = np.array([-1, -1])
    annotator_estimates[1, :, :] = np.array([1, 3])
    annotator_estimates[2:, :] = np.array([2.1, 5])

    mean = np.array([[2, 4]])
    expected_precision = np.array([[0.33333, 0.2], [1, 1], [10, 1]])

    return annotator_estimates, mean, expected_precision


def test_annotator_precision(estimated_landmarks):
    """Test compute annotator precision"""

    estimated_landmarks, mean, expected_precision = estimated_landmarks

    precision = annotator_precision(estimated_landmarks, mean)

    np.testing.assert_almost_equal(precision, expected_precision, decimal=2)


def test_for_div_zero(estimated_landmarks):
    """Test for divide by zero error"""

    estimated_landmarks, mean, expected_precision = estimated_landmarks

    precision = annotator_precision(estimated_landmarks, mean)

    assert not np.any(np.isinf(precision))


def test_select_landmarks():
    """Test selecting annotators from precision"""

    precision = np.array([[0.33333, 0.2], [1, 1.2], [10, 1.3], [5, 6]])
    landmarks = np.ones((4, 2, 2))

    landmarks, (included, excluded) = select_landmarks(precision, landmarks)

    assert landmarks.shape == (3, 2, 2)

    np.testing.assert_equal(included, [1, 3, 2])

    np.testing.assert_equal(excluded, [0])


def test_converge_mean(estimated_landmarks):
    """Test converging on a mean value"""

    estimated_landmarks, mean, estimated_precision = estimated_landmarks

    global_mean, precision = converge_mean(estimated_landmarks, iterations=1)

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(global_mean, mean)

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(precision, estimated_precision)


def test_converge_stopping(estimated_landmarks):
    """Test stopping by mean convergence"""

    estimated_landmarks, mean, estimated_precision = estimated_landmarks

    with patch('johnstondechazal.method.annotator_precision',
               return_value=estimated_precision) as p_mock:

        global_mean, precision = converge_mean(estimated_landmarks,
                                               iterations=100)

        assert p_mock.call_count == 2
