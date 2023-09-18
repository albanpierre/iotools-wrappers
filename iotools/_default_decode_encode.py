
from io import StringIO, BytesIO


def _default_decode_str(func, data, **kwargs):
    filelike = StringIO(data)
    ret = func(filelike, **kwargs)
    filelike.close()
    return ret


def _default_encode_str(func, data, **kwargs):
    filelike = StringIO()
    func(filelike, data, **kwargs)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def _default_decode_bytes(func, data, **kwargs):
    filelike = BytesIO(data)
    ret = func(filelike, **kwargs)
    filelike.close()
    return ret


def _default_encode_bytes(func, data, **kwargs):
    filelike = BytesIO()
    func(filelike, data, **kwargs)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def _default_encode_bytes_or_str(func, data, **kwargs):
    try:
        ret = _default_encode_bytes(func, data, **kwargs)
    except TypeError as e:
        if e.args[0] == "a bytes-like object is required, not 'str'":
            ret = _default_encode_str(func, data, **kwargs)
    return ret
