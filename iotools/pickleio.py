"""
This module provides io for pickle files (.pickle extension)
See iotools.pickleio.help_pickle() for more info
"""

import pickle

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_pickle(data, **kwargs):
    """
    Decode pickle data from bytes
    Backend used: pickle.loads (pickle package)

    Args:
        data (bytes): pickle data to decode
        **kwargs: same as in 'pickle.loads'

    Returns:
        pickle data decoded
    """
    return pickle.loads(data, **kwargs)


def encode_pickle(data, **kwargs):
    """
    Encode pickle data to bytes
    Backend used: pickle.dumps (pickle package)

    Args:
        data: pickle data to encode
        **kwargs: same as in 'pickle.dumps'

    Returns:
        bytes: pickle data encoded
    """
    return pickle.dumps(data, **kwargs)


def read_pickle(openfile, **kwargs):
    """
    Read a open pickle file
    Backend used: pickle.loads (pickle package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'pickle.loads'

    Returns:
        pickle data read from the file provided
    """
    # return pickle.load(openfile, **kwargs) # does not work when combined with zstd
    data = openfile.read()
    return pickle.loads(data, **kwargs)


def write_pickle(openfile, data, **kwargs):
    """
    Write in a open pickle file
    Backend used: pickle.dumps (pickle package)

    Args:
        openfile (file-like): file to write, must have been opened
        data: pickle data to save
        **kwargs: same as in 'pickle.dump'
    """
    return pickle.dump(data, openfile, **kwargs)


def load_pickle(filename, fs=None, **kwargs):
    """
    Load a pickle file
    Backend used: pickle.loads (pickle package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'pickle.loads'

    Returns:
        pickle data loaded from the file provided
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_pickle(f, **kwargs)
    return data


def save_pickle(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a pickle file
    Backend used: pickle.dumps (pickle package)

    Args:
        filename (str): file name to save data to
        data: pickle data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'pickle.dump'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_pickle(f, data, **kwargs)


def help_pickle():
    """
    Print help for pickle io
    """
    _help_default("pickle")
