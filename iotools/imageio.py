"""
This module provides io for images (.jpg, .png extensions)
See iotools.imageio.help_image() for more info
"""

import os

from iotools.settings import settings
from iotools._default_decode_encode import _default_decode_bytes, _default_encode_bytes
from iotools._generic_open import _generic_open, _generic_makedirs
from iotools._help_utils import _help_default, _help_multi_packages
from iotools._missing_module_helper import _error_msg_sev, _EmptyModule, _warning_msg_default_one
from iotools._temp_filename_utils import _get_temp_filename
from iotools.localfs import is_localfs


DEFAULT_FORMAT = "png"
AVAILABLE_FORMATS = {"jpeg", "jpg", "png"}
DEFAULT_PROPERTIES = {
    "jpg": {"quality": 90},
}


class _Image(dict):
    pass


try:
    import numpy as np
except ModuleNotFoundError:
    from iotools._missing_module_helper import _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.imageio", "numpy"))
    np = _EmptyModule("numpy")


_import_ok = set()

try:
    import imageio
    _import_ok.add("imageio")
except ModuleNotFoundError:
    imageio = _EmptyModule("imageio")

try:
    import skimage.io as skimageio
    _import_ok.add("skimage")
except ModuleNotFoundError:
    skimageio = _EmptyModule("scikit-image")

try:
    import PIL.Image as PilImage
    _import_ok.add("PilImage")
except ModuleNotFoundError:
    PilImage = _EmptyModule("Pillow")

try:
    import cv2
    _import_ok.add("cv2")
except ModuleNotFoundError:
    cv2 = _EmptyModule("opencv-python")


if len(_import_ok) == 0:
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def _guess_format_from_filename(filename, default_format=DEFAULT_FORMAT):
    """
    Guess format from filename

    Args:
        filename (str | *): filename
        default_format (str): default format returned if format cannot be guessed from filename

    Returns:
        format (str): format guessed from filename. Returns 'default_format' if format cannot be guessed
    """
    if isinstance(filename, str) and ("." in filename):
        fmt = filename.rsplit(".", 1)[1].lower()
        if fmt in AVAILABLE_FORMATS:
            return fmt
    return default_format


def _insert_properties_in_data(filename, data, kwargs):
    """
    Insert properties from kwargs to data
    """
    res = _Image()
    res["default_properties_used"] = set()
    if isinstance(data, dict):
        res["image"] = data.pop("image")
    else:
        res["image"] = data
        data = {}
    # handle format
    if "format" in data:
        if "format" in kwargs:
            print("Warning: format provided twice.")
            kwargs.pop("format")
        res["format"] = data.pop("format")
    elif "format" in kwargs:
        res["format"] = kwargs.pop("format")
    else:
        res["format"] = _guess_format_from_filename(filename)
    res["format"] = res["format"].strip('.').lower()
    if res["format"] == "jpeg":
        res["format"] = "jpg"
    # handle properties for each format
    for prop_name, prop_default in DEFAULT_PROPERTIES.get(res["format"], {}).items():
        if prop_name in data:
            if prop_name in kwargs:
                print("Warning: property '{}' provided twice.".format(prop_name))
                kwargs.pop(prop_name)
            res[prop_name] = data.pop(prop_name)
        elif prop_name in kwargs:
            res[prop_name] = kwargs.pop(prop_name)
        else:
            res[prop_name] = prop_default
            res["default_properties_used"].add(prop_name)
    for prop in data.keys():
        print("Warning: ignoring unknown property '{}'.".format(prop))
    return res, kwargs


def _del_alpha_channel(img):
    """
    Deletes the alpha channel
    """
    if isinstance(img, np.ndarray) and (img.ndim == 3) and (img.shape[2] == 4):
        alpha = set(np.unique(img[:, :, 3]).tolist())
        if alpha != {255}:
            print("Warning: 'jpg' format does not support alpha channel. Ignoring alpha channel.")
        return img[:, :, :3]
    return img


# +---------------+
# | Using imageio |
# +---------------+


def decode_image_using_imageio(data, **kwargs):
    """
    Decode image data from bytes
    Backend used: imageio.v3.imread (imageio package)

    Args:
        data (bytes): image data to decode
        **kwargs: same as in 'imageio.v3.imread'

    Returns:
        array: image data decoded
    """
    return _default_decode_bytes(read_image_using_imageio, data, **kwargs)


def encode_image_using_imageio(data, **kwargs):
    """
    Encode image data to bytes
    Backend used: imageio.imsave (imageio package)

    Args:
        data (array): image data to encode
        **kwargs: same as in 'imageio.imsave'

    Returns:
        bytes: image data encoded
    """
    return _default_encode_bytes(write_image_using_imageio, data, **kwargs)


def read_image_using_imageio(openfile, **kwargs):
    """
    Read a open image file
    Backend used: imageio.v3.imread (imageio package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'imageio.v3.imread'

    Returns:
        array: image data read from the file provided
    """
    kwargs.pop("format", None)
    img = imageio.v3.imread(openfile, **kwargs)
    return img


def write_image_using_imageio(openfile, data, **kwargs):
    """
    Write in a open image file
    Backend used: imageio.imsave (imageio package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (array): image data to save
        **kwargs: same as in 'imageio.imsave'
    """
    if not isinstance(data, _Image):
        data, kwargs = _insert_properties_in_data(openfile, data, kwargs)
    if data["format"] == "jpg":
        image = _del_alpha_channel(data["image"])
        imageio.imsave(openfile, image, format=data["format"], quality=data["quality"], **kwargs)
    else:
        imageio.imsave(openfile, data["image"], format=data["format"], **kwargs)


def load_image_using_imageio(filename, fs=None, **kwargs):
    """
    Load a image file
    Backend used: imageio.v3.imread (imageio package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'imageio.v3.imread'

    Returns:
        array: loaded data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_image_using_imageio(f, **kwargs)
    return data


def save_image_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a image file
    Backend used: imageio.imsave (imageio package)

    Args:
        filename (str): file name to save data to
        data (array): image data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'imageio.imsave'
    """
    data, kwargs = _insert_properties_in_data(filename, data, kwargs)
    # When file is opened, image saver cannot infer the format from the file name
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_image_using_imageio(f, data, **kwargs)


def help_image_using_imageio():
    """
    Print help for image io using imageio library
    """
    _help_default("image_using_imageio")


# +---------------+
# | Using skimage |
# +---------------+


def decode_image_using_skimage(data, **kwargs):
    """
    Decode image data from bytes
    Backend used: skimage.io.imread (scikit-image package)

    Args:
        data (bytes): image data to decode
        **kwargs: same as in 'skimage.io.imread'

    Returns:
        array: image data decoded
    """
    return _default_decode_bytes(read_image_using_skimage, data, **kwargs)


def encode_image_using_skimage(data, **kwargs):
    """
    Encode image data to bytes
    Backend used: skimage.io.imsave (scikit-image package)

    Args:
        data (array): image data to encode
        **kwargs: same as in 'skimage.io.imsave'

    Returns:
        bytes: image data encoded
    """
    return _default_encode_bytes(write_image_using_skimage, data, **kwargs)


def read_image_using_skimage(openfile, **kwargs):
    """
    Read a open image file
    Backend used: skimage.io.imread (scikit-image package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'skimage.io.imread'

    Returns:
        array: image data read from the file provided
    """
    kwargs.pop("format", None)
    img = skimageio.imread(openfile, **kwargs)
    return img


def write_image_using_skimage(openfile, data, **kwargs):
    """
    Write in a open image file
    Backend used: skimage.io.imsave (scikit-image package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (array): image data to save
        **kwargs: same as in 'skimage.io.imsave'
    """
    if not isinstance(data, _Image):
        data, kwargs = _insert_properties_in_data(openfile, data, kwargs)
    if data["format"] == "jpg":
        image = _del_alpha_channel(data["image"])
        skimageio.imsave(openfile, image, extension="." + data["format"], quality=data["quality"], **kwargs)
    else:
        skimageio.imsave(openfile, data["image"], extension="." + data["format"], **kwargs)


def load_image_using_skimage(filename, fs=None, **kwargs):
    """
    Load a image file
    Backend used: skimage.io.imread (scikit-image package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'skimage.io.imread'

    Returns:
        array: loaded data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_image_using_skimage(f, **kwargs)
    return data


def save_image_using_skimage(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a image file
    Backend used: skimage.io.imsave (scikit-image package)

    Args:
        filename (str): file name to save data to
        data (array): image data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'skimage.io.imsave'
    """
    data, kwargs = _insert_properties_in_data(filename, data, kwargs)
    # When file is opened, image saver cannot infer the format from the file name
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_image_using_skimage(f, data, **kwargs)


def help_image_using_skimage():
    """
    Print help for image io using skimage library
    """
    _help_default("image_using_skimage")


# +-----------+
# | Using PIL |
# +-----------+


def decode_image_using_pil(data, **kwargs):
    """
    Decode image data from bytes
    Backend used: PIL.Image.open (Pillow package)

    Args:
        data (bytes): image data to decode
        **kwargs: same as in 'PIL.Image.open'

    Returns:
        array: image data decoded
    """
    return _default_decode_bytes(read_image_using_pil, data, **kwargs)


def encode_image_using_pil(data, **kwargs):
    """
    Encode image data to bytes
    Backend used: PIL.Image.save (Pillow package)

    Args:
        data (array): image data to encode
        **kwargs: same as in 'PIL.Image.save'

    Returns:
        bytes: image data encoded
    """
    return _default_encode_bytes(write_image_using_pil, data, **kwargs)


def read_image_using_pil(openfile, **kwargs):
    """
    Read a open image file
    Backend used: PIL.Image.open (Pillow package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'PIL.Image.open'

    Returns:
        array: image data read from the file provided
    """
    kwargs.pop("format", None)
    image = PilImage.open(openfile, **kwargs)
    image.load()
    return image


def write_image_using_pil(openfile, data, **kwargs):
    """
    Write in a open image file
    Backend used: PIL.Image.save (Pillow package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (array): image data to save
        **kwargs: same as in 'PIL.Image.save'
    """
    if not isinstance(data, _Image):
        data, kwargs = _insert_properties_in_data(openfile, data, kwargs)
    if isinstance(data, np.ndarray):
        data["image"] = PilImage.fromarray(data["image"])
    if data["format"] == "jpg":
        data["format"] = "jpeg"
        data["image"].save(openfile, format=data["format"], quality=data["quality"], **kwargs)
    else:
        data["image"].save(openfile, format=data["format"], **kwargs)


def load_image_using_pil(filename, fs=None, **kwargs):
    """
    Load a image file
    Backend used: PIL.Image.open (Pillow package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'PIL.Image.open'

    Returns:
        array: loaded data
    """
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_image_using_pil(f, **kwargs)
    return data


def save_image_using_pil(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a image file
    Backend used: PIL.Image.save (Pillow package)

    Args:
        filename (str): file name to save data to
        data (array): image data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'PIL.Image.save'
    """
    data, kwargs = _insert_properties_in_data(filename, data, kwargs)
    # When file is opened, image saver cannot infer the format from the file name
    with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
        write_image_using_pil(f, data, **kwargs)


def help_image_using_pil():
    """
    Print help for image io using PIL library
    """
    _help_default("image_using_pil")


# +-----------+
# | Using cv2 |
# +-----------+


def decode_image_using_cv2(data, **kwargs):
    """
    Decode image data from bytes
    Backend used: cv2.imdecode (opencv-python package)

    Args:
        data (bytes): image data to decode
        **kwargs: same as in 'cv2.imdecode'

    Returns:
        array: image data decoded
    """
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED, **kwargs)


def encode_image_using_cv2(data, **kwargs):
    """
    Encode image data to bytes
    Backend used: cv2.imencode (opencv-python package)

    Args:
        data (array): image data to encode
        **kwargs: same as in 'cv2.imencode'

    Returns:
        bytes: image data encoded
    """
    if not isinstance(data, _Image):
        data, kwargs = _insert_properties_in_data(None, data, kwargs)
    if data["format"] == "jpg":
        data["image"] = _del_alpha_channel(data["image"])
        res = cv2.imencode("." + data["format"], data["image"], [cv2.IMWRITE_JPEG_QUALITY, data["quality"]], **kwargs)
    else:
        res = cv2.imencode("." + data["format"], data["image"], **kwargs)
    return res[1].tobytes()


def read_image_using_cv2(openfile, **kwargs):
    """
    Read a open image file
    Backend used: cv2.imdecode (opencv-python package)

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'cv2.imdecode'

    Returns:
        array: image data read from the file provided
    """
    data_encoded = openfile.read()
    return decode_image_using_cv2(data_encoded, **kwargs)


def write_image_using_cv2(openfile, data, **kwargs):
    """
    Write in a open image file
    Backend used: cv2.imencode (opencv-python package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (array): image data to save
        **kwargs: same as in 'cv2.imencode'
    """
    data_encoded = encode_image_using_cv2(data, **kwargs)
    openfile.write(data_encoded)


def load_image_using_cv2(filename, fs=None, **kwargs):
    """
    Load a image file
    Backend used: cv2.imread (opencv-python package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'cv2.imread'

    Returns:
        array: loaded data
    """
    if is_localfs(fs):
        kwargs.pop("format", None)
        data = cv2.imread(filename, **kwargs)
    else:
        with _generic_open(filename, 'rb', fs=fs) as f:
            data = read_image_using_cv2(f, **kwargs)
    return data


def save_image_using_cv2(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a image file
    Backend used: cv2.imwrite (opencv-python package)

    Args:
        filename (str): file name to save data to
        data (array): image data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'cv2.imwrite'
    """
    data, kwargs = _insert_properties_in_data(filename, data, kwargs)
    # When file is opened, image saver cannot infer the format from the file name
    if is_localfs(fs):
        if makedirs:
            _generic_makedirs(filename, fs=fs)
        if filename.endswith("." + data["format"]):
            temp_filename = filename
        else:
            temp_filename = _get_temp_filename(data["format"])
        if data["format"] == "jpg":
            data["image"] = _del_alpha_channel(data["image"])
            cv2.imwrite(temp_filename, data["image"], [cv2.IMWRITE_JPEG_QUALITY, data["quality"]], **kwargs)
        else:
            cv2.imwrite(temp_filename, data["image"], **kwargs)
        if temp_filename != filename:
            os.rename(temp_filename, filename)
    else:
        with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
            write_image_using_cv2(f, data, **kwargs)


def help_image_using_cv2():
    """
    Print help for image io using cv2 library
    """
    _help_default("image_using_cv2")


# +------------------+
# | Default behavior |
# +------------------+


def decode_image(data, **kwargs):
    """
    Decode image data from bytes
    It tries the following libraries (in the same order):
        imageio.imread
        skimage.io.imread
        PIL.Image.open
        cv2.imread
    If 'imageio.imread' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        data (bytes): image data to decode
        **kwargs: same as in 'imageio.imread'

    Returns:
        array: image data decoded
    """
    if not isinstance(imageio, _EmptyModule):
        return decode_image_using_imageio(data, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("decode_image", "imageio", "decode_image_using_skimage"))
        return decode_image_using_skimage(data, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("decode_image", "imageio", "decode_image_using_pil"))
        return decode_image_using_pil(data, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("decode_image", "imageio", "decode_image_using_cv2"))
        return decode_image_using_cv2(data, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def encode_image(data, **kwargs):
    """
    Encode image data to bytes
    It tries the following libraries (in the same order):
        imageio.imsave
        skimage.io.imsave
        PIL.Image.save
        cv2.imwrite
    If 'imageio.imsave' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        data (array): image data to encode
        **kwargs: same as in 'imageio.imsave'

    Returns:
        bytes: image data encoded
    """
    if not isinstance(imageio, _EmptyModule):
        return encode_image_using_imageio(data, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("encode_image", "imageio", "encode_image_using_skimage"))
        return encode_image_using_skimage(data, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("encode_image", "imageio", "encode_image_using_pil"))
        return encode_image_using_pil(data, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("encode_image", "imageio", "encode_image_using_cv2"))
        return encode_image_using_cv2(data, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def read_image(openfile, **kwargs):
    """
    Read a open image file
    It tries the following libraries (in the same order):
        imageio.imread
        skimage.io.imread
        PIL.Image.open
        cv2.imread
    If 'imageio.imread' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        openfile (file-like): file to read, must have been opened
        **kwargs: same as in 'imageio.imread'

    Returns:
        array: image data read from the file provided
    """
    if not isinstance(imageio, _EmptyModule):
        return read_image_using_imageio(openfile, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("read_image", "imageio", "read_image_using_skimage"))
        return read_image_using_skimage(openfile, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("read_image", "imageio", "read_image_using_pil"))
        return read_image_using_pil(openfile, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("read_image", "imageio", "read_image_using_cv2"))
        return read_image_using_cv2(openfile, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def write_image(openfile, data, **kwargs):
    """
    Write in a open image file
    It tries the following libraries (in the same order):
        imageio.imsave
        skimage.io.imsave
        PIL.Image.save
        cv2.imwrite
    If 'imageio.imsave' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        openfile (file-like): file to write, must have been opened
        data (array): image data to save
        **kwargs: same as in 'imageio.imsave'
    """
    if not isinstance(imageio, _EmptyModule):
        return write_image_using_imageio(openfile, data, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("write_image", "imageio", "write_image_using_skimage"))
        return write_image_using_skimage(openfile, data, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("write_image", "imageio", "write_image_using_pil"))
        return write_image_using_pil(openfile, data, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("write_image", "imageio", "write_image_using_cv2"))
        return write_image_using_cv2(openfile, data, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def load_image(filename, fs=None, **kwargs):
    """
    Load a image file
    It tries the following libraries (in the same order):
        imageio.imread
        skimage.io.imread
        PIL.Image.open
        cv2.imread
    If 'imageio.imread' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        **kwargs: same as in 'imageio.imread'

    Returns:
        array: loaded data
    """
    if not isinstance(imageio, _EmptyModule):
        return load_image_using_imageio(filename, fs=fs, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("load_image", "imageio", "load_image_using_skimage"))
        return load_image_using_skimage(filename, fs=fs, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("load_image", "imageio", "load_image_using_pil"))
        return load_image_using_pil(filename, fs=fs, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("load_image", "imageio", "load_image_using_cv2"))
        return load_image_using_cv2(filename, fs=fs, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def save_image(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a image file
    It tries the following libraries (in the same order):
        imageio.imsave
        skimage.io.imsave
        PIL.Image.save
        cv2.imwrite
    If 'imageio.imsave' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        filename (str): file name to save data to
        data (array): image data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'imageio.imsave'
    """
    if not isinstance(imageio, _EmptyModule):
        return save_image_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    if not isinstance(skimageio, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("save_image", "imageio", "save_image_using_skimage"))
        return save_image_using_skimage(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    if not isinstance(PilImage, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("save_image", "imageio", "save_image_using_pil"))
        return save_image_using_pil(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_one("save_image", "imageio", "save_image_using_cv2"))
        return save_image_using_cv2(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    raise ModuleNotFoundError(
        _error_msg_sev("iotools.imageio", "imageio", "scikit-image", "Pillow", "opencv-python"))


def help_image():
    """
    Print help for image io using several libraries
    """
    _help_multi_packages("image", "imageio, skimage, pil, cv2")
