"""
This module provides io for jpg files (.jpg extension)
See iotools.jpgio.help_jpg() for more info
"""

from iotools._help_utils import _help_default, _help_multi_packages
from iotools.imageio import (
    decode_image_using_imageio, encode_image_using_imageio,
    read_image_using_imageio, write_image_using_imageio,
    load_image_using_imageio, save_image_using_imageio,
    decode_image_using_skimage, encode_image_using_skimage,
    read_image_using_skimage, write_image_using_skimage,
    load_image_using_skimage, save_image_using_skimage,
    decode_image_using_pil, encode_image_using_pil,
    read_image_using_pil, write_image_using_pil,
    load_image_using_pil, save_image_using_pil,
    decode_image_using_cv2, encode_image_using_cv2,
    read_image_using_cv2, write_image_using_cv2,
    load_image_using_cv2, save_image_using_cv2,
    decode_image, encode_image, read_image, write_image, load_image, save_image
)

try:
    import imageio
except ModuleNotFoundError:
    from iotools._missing_module_helper import _EmptyModule
    imageio = _EmptyModule("imageio")


# +---------------+
# | Using imageio |
# +---------------+


def decode_jpg_using_imageio(data, **kwargs):
    return decode_image_using_imageio(data, **kwargs)


def encode_jpg_using_imageio(data, **kwargs):
    kwargs["format"] = "jpg"
    return encode_image_using_imageio(data, **kwargs)


def read_jpg_using_imageio(openfile, **kwargs):
    return read_image_using_imageio(openfile, **kwargs)


def write_jpg_using_imageio(openfile, data, **kwargs):
    kwargs["format"] = "jpg"
    return write_image_using_imageio(openfile, data, **kwargs)


def load_jpg_using_imageio(filename, fs=None, **kwargs):
    return load_image_using_imageio(filename, fs=fs, **kwargs)


def save_jpg_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "jpg"
    return save_image_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_jpg_using_imageio.__doc__ = decode_image_using_imageio.__doc__.replace(" image ", " jpg image ")
encode_jpg_using_imageio.__doc__ = encode_image_using_imageio.__doc__.replace(" image ", " jpg image ")
read_jpg_using_imageio.__doc__ = read_image_using_imageio.__doc__.replace(" image ", " jpg image ")
write_jpg_using_imageio.__doc__ = write_image_using_imageio.__doc__.replace(" image ", " jpg image ")
load_jpg_using_imageio.__doc__ = load_image_using_imageio.__doc__.replace(" image ", " jpg image ")
save_jpg_using_imageio.__doc__ = save_image_using_imageio.__doc__.replace(" image ", " jpg image ")


def help_jpg_using_imageio():
    """
    Print help for jpg image io using imageio library
    """
    _help_default("jpg_using_imageio")


# +---------------+
# | Using skimage |
# +---------------+


def decode_jpg_using_skimage(data, **kwargs):
    return decode_image_using_skimage(data, **kwargs)


def encode_jpg_using_skimage(data, **kwargs):
    kwargs["format"] = "jpg"
    return encode_image_using_skimage(data, **kwargs)


def read_jpg_using_skimage(openfile, **kwargs):
    return read_image_using_skimage(openfile, **kwargs)


def write_jpg_using_skimage(openfile, data, **kwargs):
    kwargs["format"] = "jpg"
    return write_image_using_skimage(openfile, data, **kwargs)


def load_jpg_using_skimage(filename, fs=None, **kwargs):
    return load_image_using_skimage(filename, fs=fs, **kwargs)


def save_jpg_using_skimage(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "jpg"
    return save_image_using_skimage(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_jpg_using_skimage.__doc__ = decode_image_using_skimage.__doc__.replace(" image ", " jpg image ")
encode_jpg_using_skimage.__doc__ = encode_image_using_skimage.__doc__.replace(" image ", " jpg image ")
read_jpg_using_skimage.__doc__ = read_image_using_skimage.__doc__.replace(" image ", " jpg image ")
write_jpg_using_skimage.__doc__ = write_image_using_skimage.__doc__.replace(" image ", " jpg image ")
load_jpg_using_skimage.__doc__ = load_image_using_skimage.__doc__.replace(" image ", " jpg image ")
save_jpg_using_skimage.__doc__ = save_image_using_skimage.__doc__.replace(" image ", " jpg image ")


def help_jpg_using_skimage():
    """
    Print help for jpg image io using skimage library
    """
    _help_default("jpg_using_skimage")


# +-----------+
# | Using PIL |
# +-----------+


def decode_jpg_using_pil(data, **kwargs):
    return decode_image_using_pil(data, **kwargs)


def encode_jpg_using_pil(data, **kwargs):
    kwargs["format"] = "jpg"
    return encode_image_using_pil(data, **kwargs)


def read_jpg_using_pil(openfile, **kwargs):
    return read_image_using_pil(openfile, **kwargs)


def write_jpg_using_pil(openfile, data, **kwargs):
    kwargs["format"] = "jpg"
    return write_image_using_pil(openfile, data, **kwargs)


def load_jpg_using_pil(filename, fs=None, **kwargs):
    return load_image_using_pil(filename, fs=fs, **kwargs)


def save_jpg_using_pil(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "jpg"
    return save_image_using_pil(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_jpg_using_pil.__doc__ = decode_image_using_pil.__doc__.replace(" image ", " jpg image ")
encode_jpg_using_pil.__doc__ = encode_image_using_pil.__doc__.replace(" image ", " jpg image ")
read_jpg_using_pil.__doc__ = read_image_using_pil.__doc__.replace(" image ", " jpg image ")
write_jpg_using_pil.__doc__ = write_image_using_pil.__doc__.replace(" image ", " jpg image ")
load_jpg_using_pil.__doc__ = load_image_using_pil.__doc__.replace(" image ", " jpg image ")
save_jpg_using_pil.__doc__ = save_image_using_pil.__doc__.replace(" image ", " jpg image ")


def help_jpg_using_pil():
    """
    Print help for jpg image io using pil library
    """
    _help_default("jpg_using_pil")


# +-----------+
# | Using cv2 |
# +-----------+


def decode_jpg_using_cv2(data, **kwargs):
    return decode_image_using_cv2(data, **kwargs)


def encode_jpg_using_cv2(data, **kwargs):
    kwargs["format"] = "jpg"
    return encode_image_using_cv2(data, **kwargs)


def read_jpg_using_cv2(openfile, **kwargs):
    return read_image_using_cv2(openfile, **kwargs)


def write_jpg_using_cv2(openfile, data, **kwargs):
    kwargs["format"] = "jpg"
    return write_image_using_cv2(openfile, data, **kwargs)


def load_jpg_using_cv2(filename, fs=None, **kwargs):
    return load_image_using_cv2(filename, fs=fs, **kwargs)


def save_jpg_using_cv2(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "jpg"
    return save_image_using_cv2(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_jpg_using_cv2.__doc__ = decode_image_using_cv2.__doc__.replace(" image ", " jpg image ")
encode_jpg_using_cv2.__doc__ = encode_image_using_cv2.__doc__.replace(" image ", " jpg image ")
read_jpg_using_cv2.__doc__ = read_image_using_cv2.__doc__.replace(" image ", " jpg image ")
write_jpg_using_cv2.__doc__ = write_image_using_cv2.__doc__.replace(" image ", " jpg image ")
load_jpg_using_cv2.__doc__ = load_image_using_cv2.__doc__.replace(" image ", " jpg image ")
save_jpg_using_cv2.__doc__ = save_image_using_cv2.__doc__.replace(" image ", " jpg image ")


def help_jpg_using_cv2():
    """
    Print help for jpg image io using cv2 library
    """
    _help_default("jpg_using_cv2")


# +------------------+
# | Default behavior |
# +------------------+


def decode_jpg(data, **kwargs):
    return decode_image(data, **kwargs)


def encode_jpg(data, **kwargs):
    kwargs["format"] = "jpg"
    return encode_image(data, **kwargs)


def read_jpg(openfile, **kwargs):
    return read_image(openfile, **kwargs)


def write_jpg(openfile, data, **kwargs):
    kwargs["format"] = "jpg"
    return write_image(openfile, data, **kwargs)


def load_jpg(filename, fs=None, **kwargs):
    return load_image(filename, fs=fs, **kwargs)


def save_jpg(filename, data, fs=None, makedirs=False, **kwargs):
    kwargs["format"] = "jpg"
    return save_image(filename, data, fs=fs, makedirs=makedirs, **kwargs)


decode_jpg.__doc__ = decode_image.__doc__.replace(" image ", " jpg image ")
encode_jpg.__doc__ = encode_image.__doc__.replace(" image ", " jpg image ")
read_jpg.__doc__ = read_image.__doc__.replace(" image ", " jpg image ")
write_jpg.__doc__ = write_image.__doc__.replace(" image ", " jpg image ")
load_jpg.__doc__ = load_image.__doc__.replace(" image ", " jpg image ")
save_jpg.__doc__ = save_image.__doc__.replace(" image ", " jpg image ")


def help_jpg():
    """
    Print help for jpg image io using several libraries
    """
    _help_multi_packages("jpg", "imageio, skimage, pil, cv2")
