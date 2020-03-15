#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test data module

"""
__author__ = 'Ben Johnston'

import os

import numpy as np
import pandas as pd
import pytest

from johnstondechazal.data import (_json_to_landmarks, dataframe_to_numpy,
                                   json_landmarks_to_dataframe,
                                   load_all_landmarks)

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def expert_landmarks():
    return os.path.join(TEST_DIR, '2.json')


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
        'workerid': ['2', '2'],
        'type': ['expert', 'expert'],
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
        'workerid': ['A304PUXIRA930J', 'A304PUXIRA930J'],
        'type': ['worker', 'worker'],
        13: [(848, 411), (964, 511)],
        63: [(601, 464), (631, 264)],
    })

    worker_result = json_landmarks_to_dataframe(worker_landmarks)

    assert np.all(worker_result == expected_result)


def test_extract_landmarks_numpy():
    """Test extracting landmarks as numpy array"""

    input_dataframe = pd.DataFrame.from_dict({
        'filename': [
            'indoor_006.png',
            'aflw__face_41556.jpg',
        ],
        'workerid': ['A304PUXIRA930J', 'A304PUXIRA930J'],
        'type': ['worker', 'worker'],
        13: [(848, 411), (964, 511)],
        63: [(601, 464), (631, 264)],
    })

    numpy_coords = dataframe_to_numpy(input_dataframe)

    expected_result = np.array([[[848, 411], [601, 464]],
                                [[964, 511], [631, 264]]])

    assert np.all(numpy_coords == expected_result)


def test_extract_worker_lmrks_numpy():
    """Test extract worker landmarks as numpy"""

    input_dataframe = pd.DataFrame.from_dict({
        'filename': [
            'indoor_006.png',
            'aflw__face_41556.jpg',
            'indoor_006.png',
            'aflw__face_41556.jpg',
        ],
        'workerid': [
            'A304PUXIRA930J',
            'A304PUXIRA930J',
            'M203XDSORB032N',
            'M203XDSORB032N',
        ],
        'type': [
            'worker',
            'worker',
            'expert',
            'expert',
        ],
        13: [
            (848, 411),
            (964, 511),
            (852, 422),
            (863, 422),
        ],
        63: [
            (601, 464),
            (631, 264),
            (621, 363),
            (633, 403),
        ],
    })

    expected_result = np.zeros((2, 2, 2, 2))

    expected_result[0] = np.array([[[848, 411], [601, 464]],
                                   [[964, 511], [631, 264]]])
    expected_result[1] = np.array([[[852, 422], [621, 363]],
                                   [[863, 422], [633, 403]]])

    numpy_coords, df_meta = dataframe_to_numpy(input_dataframe)

    assert np.all(numpy_coords == expected_result)

    expected_meta = pd.DataFrame.from_dict({
        'workerid': ['A304PUXIRA930J', 'M203XDSORB032N'],
        'type': ['worker', 'expert'],
    })


def test_load_all_landmarks():
    """Test loading all landmarks"""

    df = load_all_landmarks()

    for workerid, df_worker in df.groupby(df.workerid):
        assert len(df_worker) == 16

        for image, df_image in df_worker.groupby(df_worker.filename):
            assert len(df_image) == 4

            cols = []

            for _col in df_image.columns:

                try:
                    cols.append(int(_col))
                except ValueError:
                    pass

            assert len(cols) == 22
