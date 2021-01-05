#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Ground truth class

"""
import os
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd

from johnstondechazal.data import (LANDMARK_DIR, dataframe_to_numpy,
                                   download_data, load_all_landmarks)
from johnstondechazal.history import History
from johnstondechazal.method import (converge_mean, find_worst_sum,
                                     select_landmarks)


class FindGrouthTruth:
    """Class to find the ground truth landmark"""
    def __init__(self, data_dir: str = LANDMARK_DIR):
        """Constructor

        :param data_dir: The directory containing the facial landmark data,
            defaults to LANDMARK_DIR
        :type data_dir: str, optional
        """

        self.data_dir = data_dir
        self.download_data()

    def download_data(self) -> None:
        """Download the facial landmark data"""
        os.makedirs(self.data_dir, exist_ok=True)
        download_data(self.data_dir)

    def load_landmarks_image(
            self,
            image: str,
            type: Union[str, None] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load all landmarks and meta-data for an image

        :param image: The selected image
        :type image: str
        :return: The facial landmark information and meta data collected during
            the experiment.
        :rtype: Tuple[np.ndarray, pd.DataFrame]
        """

        df = load_all_landmarks(image=image, dirpath=self.data_dir)
        if type is not None:
            df = df.loc[df.type == type]
        return dataframe_to_numpy(df)

    def converge_select(self,
                        landmarks: np.ndarray,
                        meta: pd.DataFrame,
                        select_func: Callable = find_worst_sum) -> History:
        """Converge the mean for a landmark set by iteratively selecting the best
        annotators and recomputing the mean.

        :param landmarks: The landmark set
        :type landmarks: np.ndarray
        :param meta: The metadata
        :type meta: pd.DataFrame
        :param num_select: The number of landmarks to select each iteration
        :type num_select: float
        :return: The history information of the process
        :rtype: History
        """

        history = History(meta)

        # Add the global mean to the history
        mean = landmarks.mean(axis=0).mean(axis=0)
        history.add(mean, None, None)

        # Iterate for one less than number of annotators
        while 1:
            mean, precision = converge_mean(landmarks)
            landmarks, (inc, exc) = select_landmarks(precision, landmarks,
                                                     select_func)

            history.add(mean, landmarks, inc)

            if landmarks.shape[0] <= 1:
                return history
