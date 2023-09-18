"""
This module provides io for gif videos (.gif extension)
See iotools.gifio.help_gif() for more info
"""

from iotools._help_utils import _help_default, _help_multi_packages
from iotools.videoio import (
    decode_video_using_imageio, encode_video_using_imageio,
    read_video_using_imageio, write_video_using_imageio,
    load_video_using_imageio, save_video_using_imageio,
    decode_video, encode_video, read_video, write_video, load_video, save_video
)


# +---------------+
# | Using imageio |
# +---------------+


def decode_gif_using_imageio(data, which="video", **kwargs):
    return decode_video_using_imageio(data, which=which, **kwargs)


def encode_gif_using_imageio(data, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video_using_imageio(data, **kwargs)


def read_gif_using_imageio(openfile, which="video", **kwargs):
    return read_video_using_imageio(openfile, which=which, **kwargs)


def write_gif_using_imageio(openfile, data, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video_using_imageio(openfile, data, **kwargs)


def load_gif_using_imageio(filename, fs=None, which="video", **kwargs):
    return load_video_using_imageio(filename, fs=fs, which=which, **kwargs)


def save_gif_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_gif_using_imageio.__doc__ = decode_video_using_imageio.__doc__.replace(" video ", " gif video ")
encode_gif_using_imageio.__doc__ = encode_video_using_imageio.__doc__.replace(" video ", " gif video ")
read_gif_using_imageio.__doc__ = read_video_using_imageio.__doc__.replace(" video ", " gif video ")
write_gif_using_imageio.__doc__ = write_video_using_imageio.__doc__.replace(" video ", " gif video ")
load_gif_using_imageio.__doc__ = load_video_using_imageio.__doc__.replace(" video ", " gif video ")
save_gif_using_imageio.__doc__ = save_video_using_imageio.__doc__.replace(" video ", " gif video ")


def help_gif_using_imageio():
    """
    Print help for gif video io using imageio library
    """
    _help_default("gif_using_imageio")


# +------------------+
# | Default behavior |
# +------------------+


def decode_gif(data, which="video", **kwargs):
    return decode_video(data, which=which, **kwargs)


def encode_gif(data, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return encode_video(data, **kwargs)


def read_gif(openfile, which="video", **kwargs):
    return read_video(openfile, which=which, **kwargs)


def write_gif(openfile, data, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return write_video(openfile, data, **kwargs)


def load_gif(filename, fs=None, which="video", **kwargs):
    return load_video(filename, fs=fs, which=which, **kwargs)


def save_gif(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "gif"
    if isinstance(data, dict) and "properties" in data and "format" in data["properties"]:
        del data["properties"]["format"]
    return save_video(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_gif.__doc__ = decode_video.__doc__.replace(" video ", " gif video ")
encode_gif.__doc__ = encode_video.__doc__.replace(" video ", " gif video ")
read_gif.__doc__ = read_video.__doc__.replace(" video ", " gif video ")
write_gif.__doc__ = write_video.__doc__.replace(" video ", " gif video ")
load_gif.__doc__ = load_video.__doc__.replace(" video ", " gif video ")
save_gif.__doc__ = save_video.__doc__.replace(" video ", " gif video ")


def help_gif():
    """
    Print help for gif video io using several libraries
    """
    _help_multi_packages("gif", "imageio, cv2")
