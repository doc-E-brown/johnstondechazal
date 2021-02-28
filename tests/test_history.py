#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test history module

"""

import numpy as np
import pandas as pd

from johnstondechazal.history import History


def test_history():
    """Test adding to history"""

    meta = pd.DataFrame.from_dict({
        'Workerid': [1, 2, 3, 4],
        'type': ['w', 'e', 'w', 'e']
    })

    hist = History(meta)

    expected_means = [
        np.array([1, 2]),
        np.array([2, 3]),
        np.array([4, 5]),
    ]

    landmark_lists = [
        np.array([[3, 0], [2, 1], [1, 2]]),
        np.array([[3, 1], [1, 2]]),
        np.array([[3, 2]]),
    ]

    include_lists = [[0, 2, 3], [1, 2], [0]]

    for idx in range(3):
        hist.add(expected_means[idx], landmark_lists[idx], include_lists[idx])

    assert np.all(hist.loc == expected_means[-1])

    for idx, (mean, landmarks, included) in enumerate(hist):

        np.testing.assert_equal(mean, expected_means[idx])
        np.testing.assert_equal(landmarks, landmark_lists[idx])
        assert np.all(included == meta.iloc[include_lists[idx]])
