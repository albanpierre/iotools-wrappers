"""
This module provides io for mp4 videos (.mp4 extension)
See iotools.mp4io.help_mp4() for more info
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


def decode_mp4_using_imageio(data, which="video", **kwargs):
    return decode_video_using_imageio(data, which=which, **kwargs)


def encode_mp4_using_imageio(data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video_using_imageio(data, **kwargs)


def read_mp4_using_imageio(openfile, which="video", **kwargs):
    return read_video_using_imageio(openfile, which=which, **kwargs)


def write_mp4_using_imageio(openfile, data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video_using_imageio(openfile, data, **kwargs)


def load_mp4_using_imageio(filename, fs=None, which="video", **kwargs):
    return load_video_using_imageio(filename, fs=fs, which=which, **kwargs)


def save_mp4_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_mp4_using_imageio.__doc__ = decode_video_using_imageio.__doc__.replace(" video ", " mp4 video ")
encode_mp4_using_imageio.__doc__ = encode_video_using_imageio.__doc__.replace(" video ", " mp4 video ")
read_mp4_using_imageio.__doc__ = read_video_using_imageio.__doc__.replace(" video ", " mp4 video ")
write_mp4_using_imageio.__doc__ = write_video_using_imageio.__doc__.replace(" video ", " mp4 video ")
load_mp4_using_imageio.__doc__ = load_video_using_imageio.__doc__.replace(" video ", " mp4 video ")
save_mp4_using_imageio.__doc__ = save_video_using_imageio.__doc__.replace(" video ", " mp4 video ")


def help_mp4_using_imageio():
    """
    Print help for mp4 video io using imageio library
    """
    _help_default("mp4_using_imageio")


# +-----------+
# | Using cv2 |
# +-----------+


def decode_mp4_using_cv2(data, which="video", **kwargs):
    return decode_video_using_cv2(data, which=which, **kwargs)


def encode_mp4_using_cv2(data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video_using_cv2(data, **kwargs)


def read_mp4_using_cv2(openfile, which="video", **kwargs):
    return read_video_using_cv2(openfile, which=which, **kwargs)


def write_mp4_using_cv2(openfile, data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video_using_cv2(openfile, data, **kwargs)


def load_mp4_using_cv2(filename, fs=None, which="video", **kwargs):
    return load_video_using_cv2(filename, fs=fs, which=which, **kwargs)


def save_mp4_using_cv2(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video_using_cv2(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_mp4_using_cv2.__doc__ = decode_video_using_cv2.__doc__.replace(" video ", " mp4 video ")
encode_mp4_using_cv2.__doc__ = encode_video_using_cv2.__doc__.replace(" video ", " mp4 video ")
read_mp4_using_cv2.__doc__ = read_video_using_cv2.__doc__.replace(" video ", " mp4 video ")
write_mp4_using_cv2.__doc__ = write_video_using_cv2.__doc__.replace(" video ", " mp4 video ")
load_mp4_using_cv2.__doc__ = load_video_using_cv2.__doc__.replace(" video ", " mp4 video ")
save_mp4_using_cv2.__doc__ = save_video_using_cv2.__doc__.replace(" video ", " mp4 video ")


def help_mp4_using_cv2():
    """
    Print help for mp4 video io using cv2 library
    """
    _help_default("mp4_using_cv2")


# +------------------+
# | Default behavior |
# +------------------+


def decode_mp4(data, which="video", **kwargs):
    return decode_video(data, which=which, **kwargs)


def encode_mp4(data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video(data, **kwargs)


def read_mp4(openfile, which="video", **kwargs):
    return read_video(openfile, which=which, **kwargs)


def write_mp4(openfile, data, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video(openfile, data, **kwargs)


def load_mp4(filename, fs=None, which="video", **kwargs):
    return load_video(filename, fs=fs, which=which, **kwargs)


def save_mp4(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "mp4"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_mp4.__doc__ = decode_video.__doc__.replace(" video ", " mp4 video ")
encode_mp4.__doc__ = encode_video.__doc__.replace(" video ", " mp4 video ")
read_mp4.__doc__ = read_video.__doc__.replace(" video ", " mp4 video ")
write_mp4.__doc__ = write_video.__doc__.replace(" video ", " mp4 video ")
load_mp4.__doc__ = load_video.__doc__.replace(" video ", " mp4 video ")
save_mp4.__doc__ = save_video.__doc__.replace(" video ", " mp4 video ")


def help_mp4():
    """
    Print help for mp4 video io using several libraries
    """
    _help_multi_packages("mp4", "imageio, cv2")
