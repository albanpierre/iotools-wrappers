"""
This module provides io for tar files (.tar extension)
See iotools.tario.help_tar() for more info
"""

import tarfile
from io import BytesIO

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open


def decode_tar(data, **kwargs):
    """
    Decode tar data from bytes
    Backends used: tarfile.open, tarfile.getnames, tarfile.extractfile (tarfile package)

    Args:
        data (bytes): tar data to decode
        **kwargs: same as in 'tarfile.open'

    Returns:
        dict{filename: bytes}: tar data decoded
    """
    b = BytesIO(data)
    t = tarfile.open(fileobj=b, **kwargs)
    ret = {}
    for subfilename in t.getnames():
        subdata = t.extractfile(subfilename)
        if subdata is not None:
            ret[subfilename] = subdata.read()
    t.close()
    b.close()
    return ret


def encode_tar(data, **kwargs):
    """
    Encode tar data to bytes
    Backends used: tarfile.open, tarfile.TarInfo, tarfile.addfile (tarfile package)

    Args:
        data (dict{filename: bytes}): tar data to encode
        **kwargs: same as in 'tarfile.open'

    Returns:
        bytes: tar data encoded
    """
    b = BytesIO()
    t = tarfile.open(fileobj=b, mode="w", **kwargs)
    for subfilename, raw_data in data.items():
        ti = tarfile.TarInfo(subfilename)
        ti.size = len(raw_data)
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        t.addfile(ti, BytesIO(raw_data))
    b.seek(0)
    ret = b.read()
    t.close()
    b.close()
    return ret


def read_tar(openfile, **kwargs):
    """
    Read a open tar file
    Backends used: tarfile.open, tarfile.getnames, tarfile.extractfile (tarfile package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'tarfile.open'

    Returns:
        dict{filename: bytes}: tar data read from the file provided
    """
    return decode_tar(openfile.read(), **kwargs)


def write_tar(openfile, data, **kwargs):
    """
    Write in a open tar file
    Backends used: tarfile.open, tarfile.TarInfo, tarfile.addfile (tarfile package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict{filename: bytes}): tar data to save
        **kwargs: same as in 'tarfile.open'
    """
    openfile.write(encode_tar(data, **kwargs))


def load_tar(filename, fs=None, **kwargs):
    """
    Load a tar file
    Backends used: tarfile.open, tarfile.getnames, tarfile.extractfile (tarfile package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'tarfile.open'

    Returns:
        dict{filename: bytes}: tar data loaded from the file provided
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_tar(f, **kwargs)
    return data


def save_tar(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a tar file
    Backends used: tarfile.open, tarfile.TarInfo, tarfile.addfile (tarfile package)

    Args:
        filename (str): file name to save data to
        data (dict{filename: bytes}): tar data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'tarfile.open'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_tar(f, data, **kwargs)


def help_tar():
    """
    Print help for tar io
    """
    _help_default("tar")
