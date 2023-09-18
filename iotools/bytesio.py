"""
This module provides io for binary text files
See iotools.bytesio.help_bytes() for more info
"""

import os

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_bytes(data, **kwargs):
    """
    Decode bytes data, which is the identity function.

    Args:
        data (str | bytes): bytes data to decode
        **kwargs: same as in 'str.encode'

    Returns:
        bytes: decoded data
    """
    if isinstance(data, str):
        return data.encode(**kwargs)
    return data


def encode_bytes(data, **kwargs):
    """
    Encode bytes data, which is the identity function.

    Args:
        data (str | bytes): bytes data to encode
        **kwargs: same as in 'str.encode'

    Returns:
        bytes: encoded data
    """
    if isinstance(data, str):
        return data.encode(**kwargs)
    return data


def read_bytes(openfile, **kwargs):
    """
    Read an open binary text file
    Backend used: TextIOWrapper.read (builtin)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'TextIOWrapper.read'

    Returns:
        bytes: data read from the file provided
    """
    data = openfile.read(**kwargs)
    if (os.name == "nt"):  # On windows
        data = data.replace(b'\r\n', b'\n')  # Windows has different newlines encoding
    return data


def write_bytes(openfile, data, **kwargs):
    """
    Write binary text file
    Backend used: TextIOWrapper.write (builtin)

    Args:
        openfile (file-like): file to write, must have been opened
        data (bytes): text data to save
        **kwargs: same as in 'TextIOWrapper.write'
    """
    openfile.write(data, **kwargs)


def load_bytes(filename, fs=None, **kwargs):
    """
    Load a binary text file
    Backend used: TextIOWrapper.read (builtin)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'TextIOWrapper.read'

    Returns:
        bytes: loaded data
    """
    with _generic_open(filename, "rb", fs=fs) as f:
        data = read_bytes(f, **kwargs)
    return data


def save_bytes(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a binary text file
    Backend used: TextIOWrapper.write (builtin)

    Args:
        filename (str): file name to save data to
        data (bytes): text data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'TextIOWrapper.write'
    """
    with _generic_open(filename, "wb", fs=fs, makedirs=makedirs) as f:
        write_bytes(f, data, **kwargs)


def help_bytes():
    """
    Print help for bytes io
    """
    _help_default("bytes")
