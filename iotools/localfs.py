
import os

from fsspec.implementations.local import LocalFileSystem

localfs = LocalFileSystem()


def check_fs(fs):
    if (fs is None) or (fs is os):
        return localfs
    return fs


def is_localfs(fs):
    return (fs is None) or (fs is os) or isinstance(fs, LocalFileSystem)
