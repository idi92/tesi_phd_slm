import os
from pathlib import Path
ROOT_DIR_KEY = 'BRONTE_ROOT_DIR'


def set_data_root_dir(folder):
    os.environ[ROOT_DIR_KEY] = folder


def data_root_dir():

    try:
        return Path(os.environ[ROOT_DIR_KEY])
    except KeyError:
        import pkg_resources
        dataroot = pkg_resources.resource_filename(
            'bronte',
            'data')
        return Path(dataroot)


def snapshot_folder():
    return data_root_dir() / "snapshots"


def subaperture_set_folder():
    return data_root_dir() / "subaperture_set"


def phase_screen_folder():
    return data_root_dir() / "phase_screens"
