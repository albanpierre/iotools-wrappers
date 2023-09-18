"""
This module provides io for yaml files (.yaml extension)
See iotools.yamlio.help_yaml() for more info
"""

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open

try:
    import yaml
except ModuleNotFoundError:
    from iotools.settings import settings
    from iotools._missing_module_helper import _EmptyModule, _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.yamlio", "PyYAML"))
    yaml = _EmptyModule("PyYAML")


def decode_yaml(data, **kwargs):
    """
    Decode yaml data from str
    Backend used: yaml.safe_load (PyYAML package)

    Args:
        data (str): yaml data to decode
        **kwargs: same as in 'yaml.safe_load'

    Returns:
        dict: yaml data decoded
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return yaml.safe_load(data, **kwargs)


def encode_yaml(data, **kwargs):
    """
    Encode yaml data to str
    Backend used: yaml.safe_dump (PyYAML package)

    Args:
        data (dict): yaml data to encode
        **kwargs: same as in 'yaml.safe_dump'

    Returns:
        str: yaml data encoded
    """
    return yaml.safe_dump(data, **kwargs)


def read_yaml(openfile, **kwargs):
    """
    Read a open yaml file
    Backend used: yaml.safe_load (PyYAML package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'yaml.safe_load'

    Returns:
        dict: yaml data read from the file provided
    """
    return yaml.safe_load(openfile, **kwargs)


def write_yaml(openfile, data, **kwargs):
    """
    Write in a open yaml file
    Backend used: yaml.safe_dump (PyYAML package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict): yaml data to save
        **kwargs: same as in 'yaml.safe_dump'
    """
    return yaml.safe_dump(data, openfile, **kwargs)


def load_yaml(filename, fs=None, **kwargs):
    """
    Load a yaml file
    Backend used: yaml.safe_load (PyYAML package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'yaml.safe_load'

    Returns:
        dict: loaded yaml data
    """
    with _generic_open(filename, 'r', fs=fs) as f:
        data = read_yaml(f, **kwargs)
    return data


def save_yaml(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a yaml file
    Backend used: yaml.safe_dump (PyYAML package)

    Args:
        filename (str): file name to save data to
        data (dict): yaml data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'yaml.safe_dump'
    """
    with _generic_open(filename, 'w', fs=fs, makedirs=makedirs) as f:
        write_yaml(f, data, **kwargs)


def help_yaml():
    """
    Print help for yaml io
    """
    _help_default("yaml")
