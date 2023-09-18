"""
This module provides io for gzip files (.gz extension)
See iotools.gzio.help_gz() for more info
"""

import gzip

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_gz(data, **kwargs):
    """
    Decode gz data from str
    Backend used: gzip.decompress (gzip package)

    Args:
        data (bytes): gzip data to decode
        **kwargs: same as in 'gzip.decompress'

    Returns:
        bytes: gzip data decoded
    """
    return gzip.decompress(data, **kwargs)


def encode_gz(data, **kwargs):
    """
    Encode gz data to str
    Backend used: gzip.compress (gzip package)

    Args:
        data (bytes | str): gzip data to encode
        **kwargs: same as in 'gzip.compress'

    Returns:
        bytes: gzip data encoded
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return gzip.compress(data, **kwargs)


def read_gz(openfile, **kwargs):
    """
    Read a open gz file
    Backend used: gzip.decompress (gzip package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'gzip.decompress'

    Returns:
        bytes: gzip data read from the file provided
    """
    return gzip.decompress(openfile.read(), **kwargs)


def write_gz(openfile, data, **kwargs):
    """
    Write in a open gz file
    Backend used: gzip.compress (gzip package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (bytes | str): gzip data to save
        **kwargs: same as in 'gzip.compress'
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    openfile.write(gzip.compress(data, **kwargs))


def load_gz(filename, fs=None, **kwargs):
    """
    Load a gz file
    Backend used: gzip.decompress (gzip package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'gzip.decompress'

    Returns:
        bytes: loaded gzip data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_gz(f, **kwargs)
    return data


def save_gz(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a gz file
    Backend used: gzip.compress (gzip package)

    Args:
        filename (str): file name to save data to
        data (bytes | str): gzip data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'gzip.compress'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_gz(f, data, **kwargs)


def help_gz():
    """
    Print help for gzip io
    """
    _help_default("gz")
