#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

History recording class

"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class History:
    def __init__(self, meta: pd.DataFrame):
        """
        Class for storing the history of the computations

        :param meta: The meta data for the history
        :type meta: pd.DataFrame
        """
        self.records = {
            'mean': [],
            'included': [],
            'landmarks': [],
        }
        self.meta = meta.copy()

    def add(self,
            mean: np.ndarray,
            landmarks: Union[np.ndarray, None],
            include: Union[List, None] = None) -> None:
        """Add data to the history record

        :param mean: The mean to add to history
        :type mean: np.ndarray
        :param landmarks: The landmarks to include in the record
        :type landmarks: Union[np.ndarray, None]
        :param include: The indices included in the selection
        :type include: Union[List, None]
        """
        self.records['mean'].append(mean)
        self.records['landmarks'].append(landmarks)

        # Extract the record to be removed
        if include is not None:
            self.records['included'].append(self.meta.iloc[list(include)])
        else:
            self.records['included'].append(self.meta)

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        """
        for idx in range(len(self.records['mean'])):
            yield (
                self.records['mean'][idx],
                self.records['landmarks'][idx],
                self.records['included'][idx],
            )

    def __getitem__(self, key: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """[summary]

        :param key: [description]
        :type key: int
        :return: [description]
        :rtype: Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]
        """
        return (
            self.records['mean'][key],
            self.records['landmarks'][key],
            self.records['included'][key],
        )

    def __repr__(self) -> str:  # pragma : no cover
        _mean = self.records['mean'][-1]
        return f"({_mean[0]:4.2f},{_mean[1]:4.2f})@{len(self.records['mean'])}"

    def __len__(self) -> int:  # pragma: no cover
        return len(self.records['mean'])
