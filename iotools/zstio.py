"""
This module provides io for zstandard files (.zst, .zstd extensions)
See iotools.zstio.help_zst() for more info
"""

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open
from iotools._default_decode_encode import _default_decode_bytes, _default_encode_bytes

try:
    import zstandard
except ModuleNotFoundError:
    from iotools.settings import settings
    from iotools._missing_module_helper import _EmptyModule, _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.zstio", "zstandard"))
    zst = _EmptyModule("zstandard")


def decode_zst(data, **kwargs):
    """
    Decode zst data from str
    Backend used: zstandard.ZstdDecompressor (zstandard package)

    Args:
        data (bytes): data to decode
        **kwargs: same as in 'zstandard.ZstdDecompressor'

    Returns:
        bytes: zst data decoded
    """
    return _default_decode_bytes(read_zst, data, **kwargs)


def encode_zst(data, **kwargs):
    """
    Encode zst data to str
    Backend used: zstandard.ZstdCompressor (zstandard package)

    Args:
        data (bytes | str): zst data to encode
        **kwargs: same as in 'zstandard.ZstdCompressor'

    Returns:
        bytes: zst data encoded
    """
    return _default_encode_bytes(write_zst, data, **kwargs)


def read_zst(openfile, **kwargs):
    """
    Read a open zst file
    Backend used: zstandard.ZstdDecompressor (zstandard package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'zstandard.ZstdDecompressor'

    Returns:
        bytes: zst data read from the file provided
    """
    dctx = zstandard.ZstdDecompressor(**kwargs)
    reader = dctx.stream_reader(openfile)
    data = reader.read()
    return data


def write_zst(openfile, data, **kwargs):
    """
    Write in a open zst file
    Backend used: zstandard.ZstdCompressor (zstandard package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (bytes | str): zst data to save
        **kwargs: same as in 'zstandard.ZstdCompressor'
    """
    ctx = zstandard.ZstdCompressor(**kwargs)
    writer = ctx.stream_writer(openfile)
    if isinstance(data, str):
        data = data.encode('utf-8')
    writer.write(data)
    writer.flush(zstandard.FLUSH_FRAME)
    openfile.flush()


def load_zst(filename, fs=None, **kwargs):
    """
    Load a zst file
    Backend used: zstandard.ZstdDecompressor (zstandard package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'zstandard.ZstdDecompressor'

    Returns:
        bytes: loaded zst data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_zst(f, **kwargs)
    return data


def save_zst(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a zst file
    Backend used: zstandard.ZstdCompressor (zstandard package)

    Args:
        filename (str): file name to save data to
        data (bytes | str): zst data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'zstandard.ZstdCompressor'
    """
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_zst(f, data, **kwargs)


def help_zst():
    """
    Print help for zst io
    """
    _help_default("zst")
