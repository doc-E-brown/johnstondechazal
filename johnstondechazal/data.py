#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. currentmodule:: 

"""
__author__ = 'Ben Johnston'

import json
import os
import tempfile
import urllib.request
from datetime import datetime
from typing import Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd

LANDMARK_REPO = 'https://github.com/doc-E-brown/facial-landmarks/archive/master.zip'

PKG_DIR = os.path.abspath(os.path.dirname(__file__))
LANDMARK_DIR = os.path.join(PKG_DIR, 'facial-landmarks-master')


def download_data(extract_path: str = PKG_DIR) -> None:
    """
    Download facial landmark data from github into
    the package directory

    :param extract_path: The extraction path for the landmark
        data, defaults to the package path.
    :type extract_path: str

    """
    with tempfile.NamedTemporaryFile(mode='w+b') as _zip:
        urllib.request.urlretrieve(LANDMARK_REPO, _zip.name)
        _zip.seek(0)

        with ZipFile(_zip.name) as _zip_contents:
            _zip_contents.extractall(extract_path)


def _json_to_landmarks(input_json: dict) -> Tuple[str, Tuple[int, int]]:
    """Convert json to landmarks for DataFrame"""

    _id = int(input_json["id"][1:])
    return (_id, (input_json['user_x'], input_json['user_y']))


def json_landmarks_to_dataframe(filepath: str) -> Tuple[str, pd.DataFrame]:
    """ Load landmarks from a json file as pandas Dataframe """

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check if this is a MTURK or expert result
    if 'WorkerId' in data.keys():
        worker = data['WorkerId']
        data = data['Answers'][1]['FreeText']
        data = json.loads(data)
        _typ = 'worker'

    # expert result
    else:
        worker = os.path.splitext(os.path.basename(filepath))[0]
        data = data['results']
        _typ = 'expert'

    data_frame = {'filename': [], 'workerid': [], 'type': []}

    for samp in data['samples']:
        data_frame['filename'].append(os.path.basename(samp['filename']))
        data_frame['workerid'].append(worker)
        data_frame['type'].append(_typ)

        for lmrk in samp['landmarks']:
            _id, *coords = _json_to_landmarks(lmrk)

            if _id in data_frame:
                data_frame[_id].append(coords[0])
            else:
                data_frame[_id] = coords

    return pd.DataFrame.from_dict(data_frame)


def load_all_landmarks(image: Union[str, None] = None,
                       dirpath: str = LANDMARK_DIR) -> pd.DataFrame:
    """Load all the landmarks into a dataframe
    
    :param image: return landmarks for the selected image,
        defaults to `None` for all images.
    :type image: Union[str, None]
    :param dirpath: root path of all landmarks 
    :type dirpath: str
    :return: landmarks for all workers, images and replicates 
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame()
    for root, dirname, filenames in os.walk(dirpath):

        for fname in filenames:

            if '.json' not in fname:
                continue

            df = df.append(json_landmarks_to_dataframe(
                os.path.join(root, fname)),
                           ignore_index=True)

    # sort the columns
    _cols = [x for x in df.columns if isinstance(x, int)]
    _cols.sort()

    _cols = [x for x in df.columns if isinstance(x, str)] + _cols

    df = df[_cols]

    if image is not None:
        df = df.loc[df.filename == image]
        del df['filename']

    return df


def dataframe_to_numpy(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return numpy array of coordinates from a selection dataframe
    
    :param df: Input dataframe from test results
    :type df: pd.DataFrame
    :return: Selected coordinates and the metadata for the corrdinates
    :rtype: Union[np.ndarray, pd.DataFrame]
    """

    # sort the columns
    cols = [x for x in df.columns if isinstance(x, int)]
    cols.sort()

    # get the metadata colums
    cols_meta = [x for x in df.columns if isinstance(x, str)]

    # Generate array
    array = []

    df_meta = pd.DataFrame()
    for worker, worker_row in df.groupby(df.workerid):
        worker_arr = []
        df_meta = df_meta.append(
            pd.DataFrame.from_dict({
                'workerid': [worker],
                'type': [worker_row.type.unique()[0]]
            }))

        for idx, image_row in worker_row.iterrows():

            img_arr = []
            for _col in cols:
                img_arr.append(list(image_row[_col]))

            worker_arr.append(img_arr)

        array.append(worker_arr)

    array = np.array(array)

    if array.shape[0] == 1:
        return array[0]

    df_meta.index = range(len(df_meta))

    return array, df_meta


# def remove_duplicates(dirpath: str = LANDMARK_DIR):
#     """Remove duplicate workers"""

#     file_workers = {}
#     for root, dirname, filenames in os.walk(dirpath):

#         for fname in filenames:

#             if '.json' not in fname or '_' not in fname:
#                 continue

#             tstamp, worker = fname.split('_')

#             if worker not in file_workers:
#                 file_workers[worker] = []

#             file_workers[worker].append(
#                 datetime.strptime(tstamp, '%Y-%m-%d-%H:%M:%S'))

#     # Remove singulars
#     for worker in list(file_workers.keys()):

#         if len(file_workers[worker]) < 2:
#             del file_workers[worker]
#         else:
#             file_workers[worker].sort()

#             _file = file_workers[worker][0].strftime('%Y-%m-%d-%H:%M:%S')
#             _file += f'_{worker}'
#             print(_file)
#             os.remove(os.path.join(dirpath, 'landmarks', 'workers', _file))

if __name__ == "__main__":
    df = load_all_landmarks('indoor_052.png')
    arr, df_meta = dataframe_to_numpy(df)
