"""
This module provides io for avi videos (.avi extension)
See iotools.aviio.help_avi() for more info
"""

from iotools._help_utils import _help_default, _help_multi_packages
from iotools.videoio import (
    decode_video_using_imageio, encode_video_using_imageio,
    read_video_using_imageio, write_video_using_imageio,
    load_video_using_imageio, save_video_using_imageio,
    decode_video_using_cv2, encode_video_using_cv2,
    read_video_using_cv2, write_video_using_cv2,
    load_video_using_cv2, save_video_using_cv2,
    decode_video, encode_video, read_video, write_video, load_video, save_video
)


# +---------------+
# | Using imageio |
# +---------------+


def decode_avi_using_imageio(data, which="video", **kwargs):
    return decode_video_using_imageio(data, which=which, **kwargs)


def encode_avi_using_imageio(data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video_using_imageio(data, **kwargs)


def read_avi_using_imageio(openfile, which="video", **kwargs):
    return read_video_using_imageio(openfile, which=which, **kwargs)


def write_avi_using_imageio(openfile, data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video_using_imageio(openfile, data, **kwargs)


def load_avi_using_imageio(filename, fs=None, which="video", **kwargs):
    return load_video_using_imageio(filename, fs=fs, which=which, **kwargs)


def save_avi_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_avi_using_imageio.__doc__ = decode_video_using_imageio.__doc__.replace(" video ", " avi video ")
encode_avi_using_imageio.__doc__ = encode_video_using_imageio.__doc__.replace(" video ", " avi video ")
read_avi_using_imageio.__doc__ = read_video_using_imageio.__doc__.replace(" video ", " avi video ")
write_avi_using_imageio.__doc__ = write_video_using_imageio.__doc__.replace(" video ", " avi video ")
load_avi_using_imageio.__doc__ = load_video_using_imageio.__doc__.replace(" video ", " avi video ")
save_avi_using_imageio.__doc__ = save_video_using_imageio.__doc__.replace(" video ", " avi video ")


def help_avi_using_imageio():
    """
    Print help for avi video io using imageio library
    """
    _help_default("avi_using_imageio")


# +-----------+
# | Using cv2 |
# +-----------+


def decode_avi_using_cv2(data, which="video", **kwargs):
    return decode_video_using_cv2(data, which=which, **kwargs)


def encode_avi_using_cv2(data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video_using_cv2(data, **kwargs)


def read_avi_using_cv2(openfile, which="video", **kwargs):
    return read_video_using_cv2(openfile, which=which, **kwargs)


def write_avi_using_cv2(openfile, data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video_using_cv2(openfile, data, **kwargs)


def load_avi_using_cv2(filename, fs=None, which="video", **kwargs):
    return load_video_using_cv2(filename, fs=fs, which=which, **kwargs)


def save_avi_using_cv2(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video_using_cv2(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_avi_using_cv2.__doc__ = decode_video_using_cv2.__doc__.replace(" video ", " avi video ")
encode_avi_using_cv2.__doc__ = encode_video_using_cv2.__doc__.replace(" video ", " avi video ")
read_avi_using_cv2.__doc__ = read_video_using_cv2.__doc__.replace(" video ", " avi video ")
write_avi_using_cv2.__doc__ = write_video_using_cv2.__doc__.replace(" video ", " avi video ")
load_avi_using_cv2.__doc__ = load_video_using_cv2.__doc__.replace(" video ", " avi video ")
save_avi_using_cv2.__doc__ = save_video_using_cv2.__doc__.replace(" video ", " avi video ")


def help_avi_using_cv2():
    """
    Print help for avi video io using cv2 library
    """
    _help_default("avi_using_cv2")


# +------------------+
# | Default behavior |
# +------------------+


def decode_avi(data, which="video", **kwargs):
    return decode_video(data, which=which, **kwargs)


def encode_avi(data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video(data, **kwargs)


def read_avi(openfile, which="video", **kwargs):
    return read_video(openfile, which=which, **kwargs)


def write_avi(openfile, data, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video(openfile, data, **kwargs)


def load_avi(filename, fs=None, which="video", **kwargs):
    return load_video(filename, fs=fs, which=which, **kwargs)


def save_avi(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "avi"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_avi.__doc__ = decode_video.__doc__.replace(" video ", " avi video ")
encode_avi.__doc__ = encode_video.__doc__.replace(" video ", " avi video ")
read_avi.__doc__ = read_video.__doc__.replace(" video ", " avi video ")
write_avi.__doc__ = write_video.__doc__.replace(" video ", " avi video ")
load_avi.__doc__ = load_video.__doc__.replace(" video ", " avi video ")
save_avi.__doc__ = save_video.__doc__.replace(" video ", " avi video ")


def help_avi():
    """
    Print help for avi video io using several libraries
    """
    _help_multi_packages("avi", "imageio, cv2")
