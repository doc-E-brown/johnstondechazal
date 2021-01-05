#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test finding the ground truth

"""

from tempfile import mkdtemp
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from johnstondechazal.groundtruth import FindGrouthTruth


@patch('johnstondechazal.groundtruth.download_data')
def test_download_data(download_patch):
    """Test downloading data"""

    data_dir = mkdtemp()
    FindGrouthTruth(data_dir)

    download_patch.assert_called_with(data_dir)


@patch('johnstondechazal.groundtruth.download_data')
def test_load_landmarks_image(download_patch):
    """Test loading landmarks by image"""

    gt = FindGrouthTruth()
    landmarks = gt.load_landmarks_image('i001qa-mn.jpg')

    assert landmarks[0].shape == (112, 4, 22, 2)

    landmarks = gt.load_landmarks_image('i001qa-mn.jpg', 'expert')
    assert landmarks[0].shape == (12, 4, 22, 2)


@patch('johnstondechazal.groundtruth.download_data')
def test_converge_select(download_patch):
    """[summary]
    """

    gt = FindGrouthTruth()

    meta = pd.DataFrame.from_dict({
        'Workerid': ['1', '2', '3'],
        'type': ['worker', 'worker', 'expert'],
    })

    # Synthetic ground truth mean
    synth_mean = np.array([10, 12])
    samples = 4
    landmarks = np.zeros((len(meta['Workerid']), samples, 2))

    # Create the expert annotations
    expert_var = (2, 1)
    expert_x = np.random.randn(samples) * expert_var[0]
    expert_x = expert_x.astype(np.uint) + synth_mean[0]
    expert_y = np.random.randn(samples) * expert_var[1]
    expert_y = expert_y.astype(np.uint) + synth_mean[1]

    landmarks[-1, :, 0] = expert_x
    landmarks[-1, :, 1] = expert_y

    # Create the worker annotations
    worker_var = (7, 5)

    for idx in range(landmarks.shape[0] - 1):
        worker_mean = np.random.randint(5, 15, size=(2))
        worker_x = np.random.randn(samples) * worker_var[0]
        worker_x = worker_x.astype(np.uint) + worker_mean[0]
        worker_y = np.random.randn(samples) * worker_var[1]
        worker_y = worker_y.astype(np.uint) + worker_mean[1]

        landmarks[idx, :, 0] = worker_x
        landmarks[idx, :, 1] = worker_y

    gt = FindGrouthTruth()
    selection = gt.converge_select(landmarks, meta)

    global_mean = selection[0][0]
    selected_landmarks = selection[-1][0]

    assert euclidean(selected_landmarks, synth_mean) < euclidean(
        global_mean, synth_mean)
