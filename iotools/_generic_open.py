
import os

from iotools.localfs import check_fs


def _generic_makedirs(filename, fs=None):
    fs = check_fs(fs)
    if isinstance(filename, str) and not fs.exists(os.path.dirname(filename)):
        fs.makedirs(os.path.dirname(filename))


def _generic_open(filename, *args, fs=None, makedirs=False, **kwargs):
    """
    Unified open, whether the filename provided is a str path, or a file-like object
    It does open, fs.open or nothing depending on filename and fs provided
    Args:
        filename (str | file_like): The file to open/return
        *args: will be passed on to "open" and "fs.open"
        fs: the file system to use (default is None)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: will be passed on to "open" and "fs.open"
    Returns:
        (file-like) : opened file
    """
    fs = check_fs(fs)
    if isinstance(filename, str):
        if makedirs and not fs.exists(os.path.dirname(filename)):
            fs.makedirs(os.path.dirname(filename))
        return fs.open(filename, *args, **kwargs)
    return filename  # file is already opened
