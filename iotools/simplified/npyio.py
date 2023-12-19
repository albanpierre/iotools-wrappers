
import numpy as np  # pip install numpy
from io import BytesIO


def decode_npy(data, allow_pickle=False):
    filelike = BytesIO(data)
    ret = np.load(filelike, allow_pickle=allow_pickle)
    filelike.close()
    return ret


def encode_npy(data, allow_pickle=True):
    filelike = BytesIO()
    np.save(filelike, data, allow_pickle=allow_pickle)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_npy(openfile, allow_pickle=False):
    return np.load(openfile, allow_pickle=allow_pickle)


def write_npy(openfile, data, allow_pickle=True):
    np.save(openfile, data, allow_pickle=allow_pickle)


def load_npy(filename, allow_pickle=False):
    with open(filename, 'rb') as f:
        data = np.load(f, allow_pickle=allow_pickle)
    return data


def save_npy(filename, data, allow_pickle=True):
    with open(filename, 'wb') as f:
        np.save(f, data, allow_pickle=allow_pickle)
