"""
This module provides io for numpy files (.npy extension)
See iotools.npyio.help_npy() for more info
"""

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open
from iotools._default_decode_encode import _default_decode_bytes, _default_encode_bytes

try:
    import numpy as np
except ModuleNotFoundError:
    from iotools.settings import settings
    from iotools._missing_module_helper import _EmptyModule, _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.npyio", "numpy"))
    np = _EmptyModule("numpy")


def decode_npy(data, **kwargs):
    """
    Decode npy data from bytes
    Backend used: numpy.load (numpy package)

    Args:
        data (bytes): numpy data to decode
        **kwargs: same as in 'numpy.load'

    Returns:
        numpy.ndarray: numpy data decoded
    """
    return _default_decode_bytes(np.load, data, **kwargs)


def encode_npy(data, **kwargs):
    """
    Encode npy data to bytes
    Backend used: numpy.save (numpy package)

    Args:
        data (numpy.ndarray): numpy data to encode
        **kwargs: same as in 'numpy.save'

    Returns:
        bytes: numpy data encoded
    """
    return _default_encode_bytes(np.save, data, **kwargs)


def read_npy(openfile, **kwargs):
    """
    Read a open npy file
    Backend used: numpy.load (numpy package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'numpy.load'

    Returns:
        numpy.ndarray: numpy data read from the file provided
    """
    return np.load(openfile, **kwargs)


def write_npy(openfile, data, **kwargs):
    """
    Write in a open npy file
    Backend used: numpy.save (numpy package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (numpy.ndarray): numpy data to save
        **kwargs: same as in 'numpy.save'
    """
    np.save(openfile, data, **kwargs)


def load_npy(filename, fs=None, **kwargs):
    """
    Load a npy file
    Backend used: numpy.load (numpy package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'numpy.load'

    Returns:
        numpy.ndarray: loaded numpy data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_npy(f, **kwargs)
    return data


def save_npy(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a npy file
    Backend used: numpy.save (numpy package)

    Args:
        filename (str): file name to save data to
        data (numpy.ndarray): numpy data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'numpy.save'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_npy(f, data, **kwargs)


def help_npy():
    """
    Print help for npy io
    """
    _help_default("npy")
