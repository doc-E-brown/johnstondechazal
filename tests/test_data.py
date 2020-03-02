#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test data module

"""
__author__ = 'Ben Johnston'

import os
import pytest
import pandas as pd
import numpy as np

from johnstondechazal.data import json_landmarks_to_dataframe,\
    _json_to_landmarks

TEST_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

@pytest.fixture
def expert_landmarks():
    return os.path.join(TEST_DIR, 'expert.json')

@pytest.fixture
def worker_landmarks():
    return os.path.join(TEST_DIR, 'worker.json')


def test_get_lmrks_json():
    """Test getting landmarks from a json object"""

    _lmkrs = {
        "id": "P13",
        "x": 849.8,
        "y": 432.2,
        "distLim": 94.4,
        "user_x": 856,
        "user_y": 375,
        "select_time": 1540268034835
    }

    expected_result = (13, (856, 375))

    assert expected_result == _json_to_landmarks(_lmkrs)


def test_load_expert_landmarks(expert_landmarks):
    """Test loading expert landmarks"""

    expected_result = pd.DataFrame.from_dict({
        'filename': [
            'indoor_006.png',
            'aflw__face_41556.jpg',
        ],
        'workerid' : ['expert', 'expert'],
        13: [(856, 375), (956, 475)],
        61: [(456, 274), (488, 726)],
    })

    expert_result = json_landmarks_to_dataframe(expert_landmarks)

    assert np.all(expert_result == expected_result)


def test_load_worker_landmarks(worker_landmarks):
    """Test loading worker landmarks"""

    expected_result = pd.DataFrame.from_dict({
        'filename': [
            'indoor_006.png',
            'aflw__face_41556.jpg',
        ],
        'workerid' : ['A304PUXIRA930J', 'A304PUXIRA930J'],
        13: [(848, 411), (964, 511)],
        63: [(601, 464), (631, 264)],
    })

    worker_result = json_landmarks_to_dataframe(worker_landmarks)

    assert np.all(worker_result == expected_result)