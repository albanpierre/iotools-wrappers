
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.txtio import decode_txt, encode_txt
from iotools.npyio import decode_npy, encode_npy
from iotools.pngio import decode_png, encode_png
from iotools.jsonio import decode_json, encode_json
from iotools.pickleio import decode_pickle, encode_pickle
from iotools.simplified.tario import decode_tar, encode_tar, read_tar, write_tar, load_tar, save_tar


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_TAR = os.path.join(TEST_DATA_PATH, "tar", "example_tar_{}.tar")

data = {}
data[0] = {
    "example_txt_0.txt": "1, 2, 3, -67, tyu cvd",
    "example_dir_0/example_pickle_0.pickle": [1, '2', 3, -67, 'tyu'],
    "example_dir_0/example_json_0.json": [1, '2', 3, -67, 'ty'],
}
data[1] = {
    "example_dir_1/example_txt_0.txt": "1, 2, 3, -67, tyu cvd",
    "example_dir_0/example_pickle_0.pickle": [1, '2', 3, -67, 'tyu'],
    "example_dir_0/example_json_0.json": [1, '2', 3, -67, 'ty'],
    "example_npy_0.npy": [[1, 2, None], [None, "ty", 8.89]],
    "example_dir_0/example_dir_2/example_png_0.png": [[[255, 255, 255, 255],
                                                       [255, 216, 0, 255],
                                                       [255, 0, 0, 255]],
                                                      [[0, 255, 33, 255],
                                                       [0, 0, 0, 255],
                                                       [0, 255, 255, 255]]],
}


def _decode_sub_files(data):
    res = {}
    for k, v in data.items():
        if k.endswith(".txt"):
            res[k] = decode_txt(v)
        elif k.endswith(".json"):
            res[k] = decode_json(v)
        elif k.endswith(".pickle"):
            res[k] = decode_pickle(v)
        elif k.endswith(".npy"):
            res[k] = decode_npy(v, allow_pickle=True).tolist()
        elif k.endswith(".png"):
            res[k] = decode_png(v).tolist()
        else:
            res[k] = v
    return res


def _encode_sub_files(data):
    res = {}
    for k, v in data.items():
        if k.endswith(".txt"):
            res[k] = encode_txt(v)
        elif k.endswith(".json"):
            res[k] = encode_json(v)
        elif k.endswith(".pickle"):
            res[k] = encode_pickle(v)
        elif k.endswith(".npy"):
            res[k] = encode_npy(np.asarray(v))
        elif k.endswith(".png"):
            res[k] = encode_png(np.asarray(v, dtype=np.uint8))
        else:
            res[k] = v
    return res


###


def test_help_decode_tar():
    for i in range(len(data)):
        filename = FILENAMES_TAR.format(i)
        txt = load_bytes(filename)
        ldata = decode_tar(txt)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while decoding tar file nbr {}".format(i)


def test_help_encode_tar():
    for i in range(len(data)):
        txt = encode_tar(_encode_sub_files(data[i]))
        ldata = decode_tar(txt)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while encoding tar file nbr {}".format(i)


def test_help_read_tar():
    for i in range(len(data)):
        filename = FILENAMES_TAR.format(i)
        with open(filename, "rb") as f:
            ldata = read_tar(f)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while reading tar file nbr {}".format(i)


def test_help_write_tar():
    filename = FILENAMES_TAR.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_tar(f, _encode_sub_files(data[i]))
        with open(filename, "rb") as f:
            ldata = read_tar(f)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while writing tar file nbr {}".format(i)
    os.remove(filename)


def test_help_load_tar():
    for i in range(len(data)):
        filename = FILENAMES_TAR.format(i)
        ldata = load_tar(filename)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while loading tar file nbr {}".format(i)


def test_help_save_tar():
    filename = FILENAMES_TAR.format("write")
    for i in range(len(data)):
        save_tar(filename, _encode_sub_files(data[i]))
        ldata = load_tar(filename)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while saving tar file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_tar():
    save_tar("filename.tar", _encode_sub_files(data[0]))
    import iotools.examples.examples_tar  # noqa
    os.remove("filename.tar")
