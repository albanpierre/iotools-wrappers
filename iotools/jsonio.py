"""
This module provides io for json files (.json extension)
See iotools.jsonio.help_json() for more info
"""

import json

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_json(data, **kwargs):
    """
    Decode json data from str
    Backend used: json.loads (json package)

    Args:
        data (str): json data to decode
        **kwargs: same as in 'json.loads'

    Returns:
        dict | list: json data decoded
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data, **kwargs)


def encode_json(data, **kwargs):
    """
    Encode json data to str
    Backend used: json.dumps (json package)

    Args:
        data (list | dict): json data to encode
        **kwargs: same as in 'json.dumps'

    Returns:
        str: json data encoded
    """
    return json.dumps(data, **kwargs)


def read_json(openfile, **kwargs):
    """
    Read a open json file
    Backend used: json.load (json package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'json.load'

    Returns:
        dict | list: data read from the file provided
    """
    return json.load(openfile, **kwargs)


def write_json(openfile, data, **kwargs):
    """
    Write in a open json file
    Backend used: json.dump (json package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (list | dict): json data to save
        **kwargs: same as in 'json.dump'
    """
    json.dump(data, openfile, **kwargs)


def load_json(filename, fs=None, **kwargs):
    """
    Load a json file
    Backend used: json.load (json package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'json.load'

    Returns:
        dict | list: loaded json data
    """
    with _generic_open(filename, 'r', fs=fs) as f:
        data = read_json(f, **kwargs)
    return data


def save_json(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a json file
    Backend used: json.dump (json package)

    Args:
        filename (str): file name to save data to
        data (list | dict): json data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'json.dump'
    """
    with _generic_open(filename, 'w', fs=fs, makedirs=makedirs) as f:
        write_json(f, data, **kwargs)


def help_json():
    """
    Print help for json io
    """
    _help_default("json")
