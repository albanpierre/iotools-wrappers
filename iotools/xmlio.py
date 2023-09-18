"""
This module provides io for xml files (.xml extension)
See iotools.xmlio.help_xml() for more info
"""

from iotools._help_utils import _help_default
from iotools._generic_open import _generic_open

try:
    import xmltodict
except ModuleNotFoundError:
    from iotools.settings import settings
    from iotools._missing_module_helper import _EmptyModule, _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.xmlio", "xmltodict"))
    xmltodict = _EmptyModule("xmltodict")


def decode_xml(data, **kwargs):
    """
    Decode xml data from str
    Backend used: xmltodict.parse (xmltodict package)

    Args:
        data (str): xml data to decode
        **kwargs: same as in 'xmltodict.parse'

    Returns:
        OrderedDict: xml data decoded
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return xmltodict.parse(data, **kwargs)


def encode_xml(data, **kwargs):
    """
    Encode xml data to str
    Backend used: xmltodict.unparse (xmltodict package)

    Args:
        data (OrderedDict | dict): xml data to encode
        **kwargs: same as in 'xmltodict.unparse'

    Returns:
        str: xml data encoded
    """
    return xmltodict.unparse(data, **kwargs)


def read_xml(openfile, **kwargs):
    """
    Read an open xml file
    Backend used: xmltodict.parse (xmltodict package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'xmltodict.parse'

    Returns:
        OrderedDict: xml data read from the file provided
    """
    data = openfile.read()
    return xmltodict.parse(data, **kwargs)


def write_xml(openfile, data, **kwargs):
    """
    Write in an open xml file
    Backend used: xmltodict.unparse (xmltodict package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (OrderedDict | dict): xml data to save
        **kwargs: same as in 'xmltodict.unparse'
    """
    data = xmltodict.unparse(data, **kwargs)
    openfile.write(data)


def load_xml(filename, fs=None, **kwargs):
    """
    Load a xml file
    Backend used: xmltodict.parse (xmltodict package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'xmltodict.parse'

    Returns:
        OrderedDict: xml data read from the file provided
    """
    with _generic_open(filename, 'r', fs=fs) as f:
        data = read_xml(f, **kwargs)
    return data


def save_xml(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a xml file
    Backend used: xmltodict.unparse (xmltodict package)

    Args:
        filename (str): file name to save data to
        data (OrderedDict | dict): xml data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'xmltodict.unparse'
    """
    with _generic_open(filename, 'w', fs=fs, makedirs=makedirs) as f:
        write_xml(f, data, **kwargs)


def help_xml():
    """
    Print help for xml io
    """
    _help_default("xml")
