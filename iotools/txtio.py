"""
This module provides io for text files (.txt extension)
See iotools.txtio.help_txt() for more info
"""

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_txt(data, **kwargs):
    """
    Decode text data
    Backend used: str.decode (builtin)

    Args:
        data (str | bytes): text data to decode
        **kwargs: same as in 'bytes.decode'

    Returns:
        str: decoded text data
    """
    if isinstance(data, bytes):
        return data.decode(**kwargs)
    return data


def encode_txt(data, **kwargs):
    """
    Encode text data
    Backend used: str.encode (builtin)

    Args:
        data (str | bytes): text data to encode
        **kwargs: same as in 'str.encode'

    Returns:
        str: encoded text data
    """
    if isinstance(data, bytes):
        return data
    return data.encode(**kwargs)


def read_txt(openfile, **kwargs):
    """
    Read an open text file
    Backend used: TextIOWrapper.read (builtin)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'TextIOWrapper.read'

    Returns:
        str: text data read from the file provided
    """
    return openfile.read(**kwargs)


def write_txt(openfile, data, **kwargs):
    """
    Write in an open text file
    Backend used: TextIOWrapper.write (builtin)

    Args:
        openfile (file-like): file to write, must have been opened
        data (str): text data to save
        **kwargs: same as in 'TextIOWrapper.write'
    """
    openfile.write(data, **kwargs)


def load_txt(filename, fs=None, **kwargs):
    """
    Load a text file
    Backend used: TextIOWrapper.read (builtin)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'TextIOWrapper.read'

    Returns:
        str: loaded text data
    """
    with _generic_open(filename, "r", fs=fs) as f:
        data = read_txt(f, **kwargs)
    return data


def save_txt(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a text file
    Backend used: TextIOWrapper.write (builtin)

    Args:
        filename (str): file name to save data to
        data (str): text data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'TextIOWrapper.write'
    """
    with _generic_open(filename, "w", fs=fs, makedirs=makedirs) as f:
        write_txt(f, data, **kwargs)


def help_txt():
    """
    Print help for txt io
    """
    _help_default("txt")
