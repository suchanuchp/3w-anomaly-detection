import os
import configparser

import raw_data

PATH_DATASET = os.path.dirname(os.path.abspath(raw_data.__file__))
PATH_DATASET_INI = os.path.join(PATH_DATASET, "dataset.ini")


def load_ini_file():
    dataset_ini = configparser.ConfigParser()
    dataset_ini.optionxform = lambda option: option
    dataset_ini.read(PATH_DATASET_INI)
    return dict(dataset_ini)


DATASET_INI = load_ini_file()
COLUMNS_DESCRIPTIONS = dict(DATASET_INI.get("Columns of CSV Data Files"))
COLUMNS_DATA_FILES = list(COLUMNS_DESCRIPTIONS.keys())
