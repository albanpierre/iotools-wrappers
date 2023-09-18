"""
This module provides io for videos (.avi, .gif and .mp4 extensions)
See iotools.videoio.help_video() for more info
"""

import os

from iotools.settings import settings
from iotools._default_decode_encode import _default_decode_bytes, _default_encode_bytes
from iotools._generic_open import _generic_open, _generic_makedirs
from iotools._help_utils import _help_default, _help_multi_packages
from iotools._missing_module_helper import _error_msg_sev, _EmptyModule, _warning_msg_default_sev
from iotools._temp_filename_utils import _get_temp_filename
from iotools.localfs import is_localfs

try:
    import numpy as np
except ModuleNotFoundError:
    from iotools._missing_module_helper import _error_msg_one
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_one("iotools.videoio", "numpy"))
    np = _EmptyModule("numpy")


_import_ok = set()

try:
    import imageio
    import imageio_ffmpeg  # noqa
    _import_ok.add("imageio")
except ModuleNotFoundError:
    imageio = _EmptyModule("imageio", "imageio-ffmpeg")

try:
    import cv2
    _import_ok.add("cv2")
except ModuleNotFoundError:
    cv2 = _EmptyModule("opencv-python")

if len(_import_ok) == 0:
    if settings["SHOW_IMPORT_WARNINGS"]:
        print("Warning:", _error_msg_sev("iotools.videoio", "imageio imageio-ffmpeg", "opencv-python"))


# +---------------+
# | Using imageio |
# +---------------+


DEFAULT_FORMAT = "mp4"
AVAILABLE_FORMATS = {"avi", "gif", "mp4"}
DEFAULT_PROPERTIES_IMAGEIO = {
    "avi": {"fps": 30},
    "gif": {"fps": 12},
    "mp4": {"fps": 30},
}
DEFAULT_PROPERTIES_CV2 = {
    "avi": {"fps": 30, "codec": "MJPG"},
    "gif": {"fps": 12, "codec": None},
    "mp4": {"fps": 30, "codec": "mp4v"},
}


class _Video(dict):
    pass


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


def _insert_properties_in_data(filename, data, kwargs, default_properties):
    """
    Insert properties from kwargs to data
    """
    res = _Video()
    prop = {}
    prop["default_properties_used"] = set()
    properties_used = set()
    if isinstance(data, dict):
        data = data.copy()
    else:
        data = {"video": data}
    for key in {"video", "audio"}:
        res[key] = data.get(key, None)
    if "properties" not in data:
        data["properties"] = {}
    # handle format
    # get format from data or kwargs
    if "format" in kwargs:
        if "format" in data["properties"]:
            print("Warning: format provided twice.")
            properties_used.add("format")
        prop["format"] = kwargs.pop("format")
    elif "format" in data["properties"]:
        prop["format"] = data["properties"]["format"]
        properties_used.add("format")
    else:
        prop["format"] = _guess_format_from_filename(filename)
    prop["format"] = prop["format"].strip('.').lower()
    # check if format is available
    if prop["format"] not in AVAILABLE_FORMATS:
        print(
            "Warning: format '{}' is unknown or unavailable yet.".format(prop["format"]),
            "Using format '{}' instead".format(DEFAULT_FORMAT)
        )
        prop["format"] = DEFAULT_FORMAT
    # handle properties for each format
    for prop_name, prop_default in default_properties.get(prop["format"], {}).items():
        if prop_name in kwargs:
            if prop_name in data["properties"]:
                print("Warning: property '{}' provided twice.".format(prop_name))
                properties_used.add(prop_name)
            prop[prop_name] = kwargs.pop(prop_name)
        elif prop_name in data["properties"]:
            prop[prop_name] = data["properties"][prop_name]
            properties_used.add(prop_name)
        else:
            prop[prop_name] = prop_default
            prop["default_properties_used"].add(prop_name)
    # for prop_name in data["properties"].keys():
    #     if prop_name not in properties_used:
    #         print("Warning: ignoring unknown property '{}'.".format(prop_name))
    res["properties"] = prop
    return res, kwargs


###


def _imageio_properties_to_format(properties):
    if properties.get("version", b"").startswith(b"GIF"):
        return "gif"
    if properties.get("codec", "").startswith("gif"):
        return "gif"
    if properties.get("codec", "").startswith("mjpeg"):
        return "avi"
    if properties.get("codec", "").startswith("mpeg4"):
        return "mp4"
    if properties.get("codec", "").startswith("h26"):
        return "mp4"
    return "unknown"


def _encode_video_using_imageio(data, kwargs):
    prop = data["properties"]
    if settings["SHOW_DISK_USE_WARNINGS"]:
        print(
            "Warning: writing a '{}' video using imageio creates a temporary file".format(prop["format"])
            + " on local hard drive (it is removed automatically)."
        )
    temp_filename = _get_temp_filename(prop["format"])
    _save_video_using_imageio(temp_filename, data, kwargs)
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def _save_video_using_imageio(filename, data, kwargs):
    prop = kwargs.copy()
    prop.update(data["properties"])
    del prop["default_properties_used"]
    if prop["format"] == "gif" and "fps" in prop:
        prop["duration"] = 1000 * 1 / prop.pop("fps")
    writer = imageio.get_writer(filename, **prop)
    for frame in data["video"]:
        writer.append_data(frame)
    writer.close()


def decode_video_using_imageio(data, which="video", **kwargs):
    """
    Decode video data from bytes
    Backend used: imageio.get_reader (imageio package)

    Args:
        data (bytes): video data to decode
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data decoded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    return _default_decode_bytes(read_video_using_imageio, data, which=which, **kwargs)


def encode_video_using_imageio(data, **kwargs):
    """
    Encode video data to bytes
    Backend used: imageio.get_writer (imageio package)

    Args:
        data (dict | np.array): video data to encode
        **kwargs: same as in 'imageio.get_writer'

    Returns:
        bytes: video data encoded
    """
    if not isinstance(data, _Video):
        data, kwargs = _insert_properties_in_data(None, data, kwargs, DEFAULT_PROPERTIES_IMAGEIO)
    if data["properties"]["format"] == "gif":
        return _default_encode_bytes(write_video_using_imageio, data, **kwargs)
    return _encode_video_using_imageio(data, kwargs)


def read_video_using_imageio(openfile, which="video", **kwargs):
    """
    Read an open video file
    Backend used: imageio.get_reader (imageio package)

    Args:
        openfile (file-like): file to read, must have been opened
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data read, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    res = {}
    # Read file and get right format
    formats = ["mp4", "avi", "gif"]
    if "format" in kwargs:
        formats = [kwargs.pop("format")] + formats
    for format in formats:
        try:
            reader = imageio.get_reader(openfile, format=format, **kwargs)
            properties = reader.get_meta_data()
            properties["format"] = _imageio_properties_to_format(properties)
            if properties["format"] != format:
                reader = imageio.get_reader(openfile, format=properties["format"], **kwargs)
                properties = reader.get_meta_data()
                properties["format"] = _imageio_properties_to_format(properties)
            break
        except ValueError:
            pass
    # Format data for user
    if ("video" in which) or ("all" in which):
        res["video"] = [im for im in reader]  # list(reader) does not work
    if ("audio" in which) or ("all" in which):
        res["audio"] = None
    if ("properties" in which) or ("all" in which):
        res["properties"] = dict(properties)
    if which == "video":
        return res["video"]
    if which == "audio":
        return res["audio"]
    if which == "properties":
        return res["properties"]
    return res


def write_video_using_imageio(openfile, data, **kwargs):
    """
    Write in an open video file
    Backend used: imageio.get_writer (imageio package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict | np.array): video data to write
        **kwargs: same as in 'imageio.get_writer'
    """
    if not isinstance(data, _Video):
        data, kwargs = _insert_properties_in_data(None, data, kwargs, DEFAULT_PROPERTIES_IMAGEIO)
    if data["properties"]["format"] == "gif":
        _save_video_using_imageio(openfile, data, kwargs)
    else:
        data_enc = _encode_video_using_imageio(data, kwargs)
        openfile.write(data_enc)


def load_video_using_imageio(filename, fs=None, which="video", **kwargs):
    """
    Load a video file
    Backend used: imageio.get_reader (imageio package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data loaded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if "format" not in kwargs:
        fmt = _guess_format_from_filename(filename, default_format=None)
        if fmt is not None:
            kwargs["format"] = fmt
    # Better to guess the format from the filename to give info to imageio
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_video_using_imageio(f, which=which, **kwargs)
    return data


def save_video_using_imageio(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a video file
    Backend used: imageio.get_writer (imageio package)

    Args:
        filename (str): file name to save data to
        data (dict | np.array): video data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'imageio.get_writer'
    """
    data, kwargs = _insert_properties_in_data(filename, data, kwargs, DEFAULT_PROPERTIES_IMAGEIO)
    if is_localfs(fs):
        if makedirs:
            _generic_makedirs(filename)
        _save_video_using_imageio(filename, data, kwargs)
    else:
        with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
            write_video_using_imageio(f, data, **kwargs)


def help_video_using_imageio():
    """
    Print help for video io using imageio library
    """
    _help_default("video_using_imageio")


# +-----------+
# | Using cv2 |
# +-----------+


def _get_video_cv2_properties(cap):
    properties = {"fps": cap.get(cv2.CAP_PROP_FPS)}
    h = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = chr(h & 0xff) + chr((h >> 8) & 0xff) + chr((h >> 16) & 0xff) + chr((h >> 24) & 0xff)
    properties["codec"] = codec.lower()
    return properties


def _cv2_properties_to_format(properties):
    if properties.get("codec", "").startswith("gif"):
        return "gif"
    if properties.get("codec", "") == "mjpg":
        return "avi"
    if properties.get("codec", "") in {"fmp4", "h264", "avc1", "mp4v"}:
        return "mp4"
    return "unknown"


def decode_video_using_cv2(data, which="video", **kwargs):
    """
    Decode video data from bytes
    Backend used: cv2.VideoCapture (opencv-python package)

    Args:
        data (bytes): video data to decode
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'cv2.VideoCapture'

    Returns:
        dict: video data decoded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if settings["SHOW_DISK_USE_WARNINGS"]:
        print(
            "Warning: reading a video using cv2 creates a temporary file on local hard drive"
            + " (it is removed automatically)."
        )
    temp_filename = _get_temp_filename()
    with open(temp_filename, "wb") as f:
        f.write(data)
    prop = {k: v for k, v in kwargs.items() if k not in {"codec", "format", "fps"}}
    data_decoded = load_video_using_cv2(temp_filename, fs=None, which=which, **prop)
    os.remove(temp_filename)
    return data_decoded


def encode_video_using_cv2(data, **kwargs):
    """
    Encode video data to bytes
    Backend used: cv2.VideoWriter (opencv-python package)

    Args:
        data (dict | np.array): video data to encode
        **kwargs: same as in 'cv2.VideoWriter'

    Returns:
        bytes: video data encoded
    """
    if not isinstance(data, _Video):
        data, kwargs = _insert_properties_in_data(None, data, kwargs, DEFAULT_PROPERTIES_CV2)
    if settings["SHOW_DISK_USE_WARNINGS"]:
        print(
            "Warning: writing a video using cv2 creates a temporary file on local hard drive"
            + " (it is removed automatically)."
        )
    temp_filename = _get_temp_filename(data["properties"]["format"])
    save_video_using_cv2(temp_filename, data, fs=None, makedirs=False, **kwargs)
    with open(temp_filename, "rb") as f:
        data_encoded = f.read()
    os.remove(temp_filename)
    return data_encoded


def read_video_using_cv2(openfile, which="video", **kwargs):
    """
    Read an open video file
    Backend used: cv2.VideoCapture (opencv-python package)

    Args:
        openfile (file-like): file to read, must have been opened
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'cv2.VideoCapture'

    Returns:
        dict: video data read, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    data_encoded = openfile.read()
    return decode_video_using_cv2(data_encoded, which=which, **kwargs)


def write_video_using_cv2(openfile, data, **kwargs):
    """
    Write in an open video file
    Backend used: cv2.VideoWriter (opencv-python package)

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict | np.array): video data to write
        **kwargs: same as in 'cv2.VideoWriter'
    """
    if not isinstance(data, _Video):
        data, kwargs = _insert_properties_in_data(None, data, kwargs, DEFAULT_PROPERTIES_CV2)
    data_encoded = encode_video_using_cv2(data, **kwargs)
    openfile.write(data_encoded)


def load_video_using_cv2(filename, fs=None, which="video", **kwargs):
    """
    Load a video file
    Backend used: cv2.VideoCapture (opencv-python package)

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'cv2.VideoCapture'

    Returns:
        dict: video data loaded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if is_localfs(fs):
        # As far as I know a local filename is required to load a video using cv2
        if "format" in kwargs:
            del kwargs["format"]
        cap = cv2.VideoCapture(filename, **kwargs)
        res = {}
        if "properties" in which or "all" in which:
            res["properties"] = _get_video_cv2_properties(cap)
            if "format" not in res["properties"]:
                res["properties"]["format"] = _cv2_properties_to_format(res["properties"])
        if "video" in which or "all" in which:
            video = []
            ret = True
            while ret:
                ret, frame = cap.read()
                video.append(frame)
            cap.release()
            res["video"] = [frame[:, :, 2::-1] for frame in video[:-1]]
        if "audio" in which or "all" in which:
            res["audio"] = None
        if which == "video":
            return res["video"]
        if which == "audio":
            return res["audio"]
        if which == "properties":
            return res["properties"]
        return res
    # If file is not on local filesystem
    with _generic_open(filename, 'rb', fs=fs) as f:
        data = read_video_using_cv2(f, which=which, **kwargs)
    return data


def save_video_using_cv2(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a video file
    Backend used: cv2.VideoWriter (opencv-python package)

    Args:
        filename (str): file name to save data to
        data (dict | np.array): video data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'cv2.VideoWriter'
    """
    if not isinstance(data, _Video):
        data, kwargs = _insert_properties_in_data(filename, data, kwargs, DEFAULT_PROPERTIES_CV2)
    if is_localfs(fs):
        if makedirs:
            _generic_makedirs(filename)
        prop = data["properties"]
        if prop["codec"] is None:
            raise ValueError("Unknown format, please provide codec")
        frames = data["video"]
        height = len(frames[0])
        width = len(frames[0][0])
        fourcc = cv2.VideoWriter_fourcc(*prop["codec"])
        out = cv2.VideoWriter(filename, fourcc, prop["fps"], (width, height), **kwargs)
        for frame in frames:
            if frame.ndim != 3:
                raise ValueError("Frames must be 2D-images.")
            if frame.shape[2] != 3:
                raise ValueError("Frames must be 2D-images with 3 color channels.")
            if "uint8" not in str(frame.dtype):
                raise ValueError("Frames must be 2D-images with 3 color channels, of type uint8")
            out.write(frame[:, :, 2::-1])
        out.release()
    else:
        with _generic_open(filename, 'wb', fs=fs, makedirs=makedirs) as f:
            write_video_using_cv2(f, data, **kwargs)


def help_video_using_cv2():
    """
    Print help for video io using cv2 library
    """
    _help_default("video_using_cv2")


# +------------------+
# | Default behavior |
# +------------------+


def decode_video(data, which="video", **kwargs):
    """
    Decode video data from bytes
    It tries the following libraries (in the same order):
        imageio.get_reader
        cv2.VideoCapture
    If 'imageio.get_reader' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        data (bytes): video data to decode
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data decoded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if not isinstance(imageio, _EmptyModule):
        return decode_video_using_imageio(data, which=which, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("decode_video", ["imageio", "imageio-ffmpeg"], "decode_video_using_cv2"))
        return decode_video_using_cv2(data, which=which, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def encode_video(data, **kwargs):
    """
    Encode video data to bytes
    It tries the following libraries (in the same order):
        imageio.get_writer
        cv2.VideoWriter
    If 'imageio.get_writer' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        data (dict | np.array): video data to encode
        **kwargs: same as in 'imageio.get_writer'

    Returns:
        bytes: video data encoded
    """
    if not isinstance(imageio, _EmptyModule):
        return encode_video_using_imageio(data, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("encode_video", ["imageio", "imageio-ffmpeg"], "encode_video_using_cv2"))
        return encode_video_using_cv2(data, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def read_video(openfile, which="video", **kwargs):
    """
    Read an open video file
    It tries the following libraries (in the same order):
        imageio.get_reader
        cv2.VideoCapture
    If 'imageio.get_reader' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        openfile (file-like): file to read, must have been opened
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data read, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if not isinstance(imageio, _EmptyModule):
        return read_video_using_imageio(openfile, which=which, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("read_video", ["imageio", "imageio-ffmpeg"], "read_video_using_cv2"))
        return read_video_using_cv2(openfile, which=which, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def write_video(openfile, data, **kwargs):
    """
    Write in an open video file
    It tries the following libraries (in the same order):
        imageio.get_writer
        cv2.VideoWriter
    If 'imageio.get_writer' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        openfile (file-like): file to write, must have been opened
        data (dict | np.array): video data to write
        **kwargs: same as in 'imageio.get_writer'
    """
    if not isinstance(imageio, _EmptyModule):
        return write_video_using_imageio(openfile, data, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("write_video", ["imageio", "imageio-ffmpeg"], "write_video_using_cv2"))
        return write_video_using_cv2(openfile, data, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def load_video(filename, fs=None, which="video", **kwargs):
    """
    Load a video file
    It tries the following libraries (in the same order):
        imageio.get_reader
        cv2.VideoCapture
    If 'imageio.get_reader' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        filename (str): file name to load
        fs: file system to use. Defaults to None (local filesystem)
        which (str | list): which data to load, either 'video', 'audio', 'properties' or 'all'.
            Can also be a list of these str (example: which=['audio', 'video']). Defaults to 'video'
        **kwargs: same as in 'imageio.get_reader'

    Returns:
        dict: video data loaded, as a dict containing the keys 'video', 'audio' and 'properties'.
            If 'which' is a single element (example: which='video'), the dict value is returned instead
    """
    if not isinstance(imageio, _EmptyModule):
        return load_video_using_imageio(filename, fs=fs, which=which, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("load_video", ["imageio", "imageio-ffmpeg"], "load_video_using_cv2"))
        return load_video_using_cv2(filename, fs=fs, which=which, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def save_video(filename, data, fs=None, makedirs=False, **kwargs):
    """
    Save a video file
    It tries the following libraries (in the same order):
        imageio.get_writer
        cv2.VideoWriter
    If 'imageio.get_writer' does not work, a warning is printed since this function may have a
    different behavior in other environments

    Args:
        filename (str): file name to save data to
        data (dict | np.array): video data to save
        fs: file system to use. Defaults to None (local filesystem)
        makedirs (bool): if True, creates directory of filename (default is False)
        **kwargs: same as in 'imageio.get_writer'
    """
    if not isinstance(imageio, _EmptyModule):
        return save_video_using_imageio(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    if not isinstance(cv2, _EmptyModule):
        if settings["SHOW_DEFAULT_BEHAVIOR_WARNINGS"]:
            print(_warning_msg_default_sev("save_video", ["imageio", "imageio-ffmpeg"], "save_video_using_cv2"))
        return save_video_using_cv2(filename, data, fs=fs, makedirs=makedirs, **kwargs)
    raise ModuleNotFoundError(_error_msg_sev("iotools.videoio", ["imageio", "imageio-ffmpeg"], "opencv-python"))


def help_video():
    """
    Print help for video io using several libraries
    """
    _help_multi_packages("video", "imageio, cv2")
