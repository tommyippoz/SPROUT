import configparser
import os
import time

import numpy as np


def load_config(file_config):
    """
    Method to load configuration parameters from input file
    :param file_config: name of the config file
    :return: array with 3 items: [dataset files, label name, max number of rows (if correctly specified, NaN otherwise)]
    """
    config = configparser.RawConfigParser()
    config.read(file_config)
    config_file = dict(config.items('CONFIGURATION'))

    # Processing paths
    path_string = config_file['path']
    if ',' in path_string:
        path_string = path_string.split(',')
    else:
        path_string = [path_string]
    datasets_path = []
    for file_string in path_string:
        if os.path.isdir(file_string):
            datasets_path.extend([os.path.join(file_string, f) for f in os.listdir(file_string) if
                                  os.path.isfile(os.path.join(file_string, f))])
        else:
            datasets_path.append(file_string)

    # Processing limit to rows
    lim_rows = config_file['limit_rows']
    if not lim_rows.isdigit():
        lim_rows = np.nan
    else:
        lim_rows = int(lim_rows)
    return datasets_path, config_file['label_tabular'], lim_rows


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def clean_name(file):
    """
    Method to get clean name of a file
    :param file: the original file path
    :return: the filename with no path and extension
    """
    name = os.path.basename(file)
    if '.' in name:
        name = name.split('.')[0]
    return name
