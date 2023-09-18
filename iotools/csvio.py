"""
This module provides io for csv files (.csv extension)
See iotools.csvio.help_csv() for more info
"""

import csv

from iotools._help_utils import _help_default, _help_multi_packages
from iotools._generic_open import _generic_open
from iotools._default_decode_encode import _default_decode_str, _default_encode_str


try:
    import pandas as pd
except ModuleNotFoundError:
    from iotools._missing_module_helper import _EmptyModule
    pd = _EmptyModule("pandas")


# +-----------+
# | Using csv |
# +-----------+


def decode_csv_using_csv(data, **kwargs):
    """
    Decode csv data from str using csv library
    Backend used: csv.reader (csv package)

    Args:
        data (str): csv data to decode
        **kwargs: same as in 'csv.reader'

    Returns:
        list: csv data decoded
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return _default_decode_str(read_csv_using_csv, data, **kwargs)


def encode_csv_using_csv(data, **kwargs):
    """
    Encode csv data to str using csv library
    Backend used: csv.writer (csv package)

    Args:
        data (list[list]): csv data to encode
        **kwargs: same as in 'csv.writer'

    Returns:
        str: csv data encoded
    """
    return _default_encode_str(write_csv_using_csv, data, **kwargs)


def read_csv_using_csv(openfile, **kwargs):
    """
    Read a open csv file using csv library
    Backend used: csv.reader (csv package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'csv.reader'

    Returns:
        list[list]: csv data read from the file provided
    """
    reader = csv.reader(openfile, **kwargs)
    return list(reader)


def write_csv_using_csv(openfile, data, **kwargs):
    """
    Write in a open csv file using csv library
    Backend used: csv.writer (csv package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (list[list]): csv data to save
        **kwargs: same as in 'csv.writer'
    """
    writer = csv.writer(openfile, **kwargs)
    for row in data:
        writer.writerow(row)


def load_csv_using_csv(filename, fs=None, **kwargs):
    """
    Load a csv file using csv library
    Backend used: csv.reader (csv package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'csv.reader'

    Returns:
        list[list]: loaded csv data
    """
    with _generic_open(filename, 'r', fs=fs) as f:
        data = read_csv_using_csv(f, **kwargs)
    return data


def save_csv_using_csv(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a csv file using csv library
    Backend used: csv.writer (csv package)

    Args:
        filename (str): file name to save data to
        data (list[list]): csv data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'csv.writer'
    """
    with _generic_open(filename, 'w', fs=fs, makedirs=makedirs) as f:
        write_csv_using_csv(f, data, **kwargs)


def help_csv_using_csv():
    """
    Print help for csv io using csv library
    """
    _help_default("csv_using_csv")


# +--------------+
# | Using pandas |
# +--------------+


def decode_csv_using_pandas(data, **kwargs):
    """
    Decode csv data from str using pandas library
    Backend used: pandas.read_csv (pandas package)

    Args:
        data (str): csv data to decode
        **kwargs: same as in 'pandas.read_csv'

    Returns:
        pandas.DataFrame: csv data decoded
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return _default_decode_str(read_csv_using_pandas, data, **kwargs)


def encode_csv_using_pandas(data, **kwargs):
    """
    Encode csv data to str using pandas library
    Backend used: pandas.DataFrame.to_csv (pandas package)

    Args:
        data (pandas.DataFrame): csv data to encode
        **kwargs: same as in 'pandas.DataFrame.to_csv'

    Returns:
        str: csv data encoded
    """
    return _default_encode_str(write_csv_using_pandas, data, **kwargs)


def read_csv_using_pandas(openfile, **kwargs):
    """
    Read a open csv file using pandas library
    Backend used: pandas.read_csv (pandas package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'pandas.read_csv'

    Returns:
        pandas.DataFrame: csv data read from the file provided
    """
    return pd.read_csv(openfile, **kwargs)


def write_csv_using_pandas(openfile, data, **kwargs):
    """
    Write in a open csv file using pandas library
    Backend used: pandas.DataFrame.to_csv (pandas package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (pandas.DataFrame): csv data to save
        **kwargs: same as in 'pandas.DataFrame.to_csv'
    """
    pd.DataFrame(data).to_csv(openfile, **kwargs)


def load_csv_using_pandas(filename, fs=None, **kwargs):
    """
    Load a csv file using pandas library
    Backend used: pandas.read_csv (pandas package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'pandas.read_csv'

    Returns:
        pandas.DataFrame: loaded csv data
    """
    with _generic_open(filename, 'r', fs=fs) as f:
        data = read_csv_using_pandas(f, **kwargs)
    return data


def save_csv_using_pandas(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a csv file using pandas library
    Backend used: pandas.DataFrame.to_csv (pandas package)

    Args:
        filename (str): file name to save data to
        data (pandas.DataFrame): csv data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'pandas.DataFrame.to_csv'
    """
    with _generic_open(filename, 'w', fs=fs, makedirs=makedirs) as f:
        write_csv_using_pandas(f, data, **kwargs)


def help_csv_using_pandas():
    """
    Print help for csv io using pandas library
    """
    _help_default("csv_using_pandas")


# +------------------+
# | Default behavior |
# +------------------+


def decode_csv(data, **kwargs):
    return decode_csv_using_csv(data, **kwargs)


def encode_csv(data, **kwargs):
    return encode_csv_using_csv(data, **kwargs)


def read_csv(openfile, **kwargs):
    return read_csv_using_csv(openfile, **kwargs)


def write_csv(openfile, data, **kwargs):
    return write_csv_using_csv(openfile, data, **kwargs)


def load_csv(filename, fs=None, **kwargs):
    return load_csv_using_csv(filename, fs=fs, **kwargs)


def save_csv(filename, data, fs=None, makedirs=False, **kwargs):
    return save_csv_using_csv(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_csv.__doc__ = decode_csv_using_csv.__doc__.replace(" using csv library", "")
encode_csv.__doc__ = encode_csv_using_csv.__doc__.replace(" using csv library", "")
read_csv.__doc__ = read_csv_using_csv.__doc__.replace(" using csv library", "")
write_csv.__doc__ = write_csv_using_csv.__doc__.replace(" using csv library", "")
load_csv.__doc__ = load_csv_using_csv.__doc__.replace(" using csv library", "")
save_csv.__doc__ = save_csv_using_csv.__doc__.replace(" using csv library", "")


def help_csv():
    """
    Print help for csv io using several libraries
    """
    _help_multi_packages("csv", "csv, pandas")
