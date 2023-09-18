"""
This module provides io for zip files (.zip extension)
See iotools.zipio.help_zip() for more info
"""

import zipfile

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open
from iotools._default_decode_encode import _default_decode_bytes, _default_encode_bytes


def decode_zip(data, **kwargs):
    """
    Decode zip data from bytes
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        data (bytes): data to decode
        **kwargs: same as in 'zipfile.ZipFile'

    Returns:
        dict{filename: bytes}: zip data decoded
    """
    return _default_decode_bytes(read_zip, data, **kwargs)


def encode_zip(data, **kwargs):
    """
    Encode zip data to bytes
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        data (dict{filename: bytes}): zip data to encode
        **kwargs: same as in 'zipfile.ZipFile'

    Returns:
        bytes: zip data encoded
    """
    return _default_encode_bytes(write_zip, data, **kwargs)


def read_zip(openfile, **kwargs):
    """
    Read a open zip file
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'zipfile.ZipFile'

    Returns:
        dict{filename: bytes}: zip data read from the file provided
    """
    z = zipfile.ZipFile(openfile, **kwargs)
    data = {fn: None for fn in z.namelist() if not fn.endswith("/")}
    for subfilename in data:
        with z.open(subfilename, "r") as subf:
            data[subfilename] = subf.read()
    z.close()
    return data


def write_zip(openfile, data, **kwargs):
    """
    Write in a open zip file
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict{filename: bytes}): zip data to save
        **kwargs: same as in 'zipfile.ZipFile'
    """
    z = zipfile.ZipFile(openfile, "w", **kwargs)
    for subfilename, raw_data in data.items():
        with z.open(subfilename, "w") as subf:
            if isinstance(raw_data, str):
                raw_data = raw_data.encode('utf-8')
            subf.write(raw_data)
    z.close()


def load_zip(filename, fs=None, **kwargs):
    """
    Load a zip file
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'zipfile.ZipFile'

    Returns:
        dict{filename: bytes}: zip data loaded from the file provided
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_zip(f, **kwargs)
    return data


def save_zip(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a zip file
    Backend used: zipfile.ZipFile (zipfile package)

    Args:
        filename (str): file name to save data to
        data (dict{filename: bytes}): zip data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'zipfile.ZipFile'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_zip(f, data, **kwargs)


def help_zip():
    """
    Print help for zip io
    """
    _help_default("zip")
