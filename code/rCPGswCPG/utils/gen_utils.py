from copy import deepcopy
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent

def get_folders(root_folder, pattern):
    folders_all = os.listdir(root_folder + '/')
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search(pattern, str(folder))
        if m is not None:
            folders.append(folder)
    return folders

def get_files(root_folder, pattern):
    folders_and_files = os.listdir(root_folder + '/')
    files = []
    for i, el in enumerate(folders_and_files):
        if os.path.isfile(root_folder  + '/' + el):
            m = re.search(pattern, str(el))
            if m is not None:
                files.append(el)
    return files

def dict_to_string(dict):
    res = ",".join(("{}={}".format(*i) for i in dict.items()))
    return res

def put(arr, ind, new_val):
    out = arr.copy()
    out[ind] = new_val
    return out

def array2str(arr):
    return np.array2string(np.array(arr), precision=6, separator='_')[1:-1]




