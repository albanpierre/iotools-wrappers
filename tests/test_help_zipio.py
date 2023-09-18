
import os
import numpy as np

from iotools.bytesio import load_bytes
from iotools.txtio import decode_txt, encode_txt
from iotools.npyio import decode_npy, encode_npy
from iotools.pngio import decode_png, encode_png
from iotools.jsonio import decode_json, encode_json
from iotools.pickleio import decode_pickle, encode_pickle
from simplified_iotools.zipio import decode_zip, encode_zip, read_zip, write_zip, load_zip, save_zip


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_ZIP = os.path.join(TEST_DATA_PATH, "zip", "example_zip_{}.zip")

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


def test_help_decode_zip():
    for i in range(len(data)):
        filename = FILENAMES_ZIP.format(i)
        txt = load_bytes(filename)
        ldata = decode_zip(txt)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while decoding zip file nbr {}".format(i)


def test_help_encode_zip():
    for i in range(len(data)):
        txt = encode_zip(_encode_sub_files(data[i]))
        ldata = decode_zip(txt)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while encoding zip file nbr {}".format(i)


def test_help_read_zip():
    for i in range(len(data)):
        filename = FILENAMES_ZIP.format(i)
        with open(filename, "rb") as f:
            ldata = read_zip(f)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while reading zip file nbr {}".format(i)


def test_help_write_zip():
    filename = FILENAMES_ZIP.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_zip(f, _encode_sub_files(data[i]))
        with open(filename, "rb") as f:
            ldata = read_zip(f)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while writing zip file nbr {}".format(i)
    os.remove(filename)


def test_help_load_zip():
    for i in range(len(data)):
        filename = FILENAMES_ZIP.format(i)
        ldata = load_zip(filename)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while loading zip file nbr {}".format(i)


def test_help_save_zip():
    filename = FILENAMES_ZIP.format("write")
    for i in range(len(data)):
        save_zip(filename, _encode_sub_files(data[i]))
        ldata = load_zip(filename)
        ldata = _decode_sub_files(ldata)
        assert ldata == data[i], "Error while saving zip file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_zip():
    save_zip("filename.zip", _encode_sub_files(data[0]))
    import examples.examples_zip  # noqa
    os.remove("filename.zip")
