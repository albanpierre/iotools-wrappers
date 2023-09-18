
import os

from iotools.bytesio import load_bytes
from iotools.jsonio import decode_json, encode_json
from iotools.pickleio import decode_pickle, encode_pickle
from iotools.zstio import decode_zst, encode_zst, read_zst, write_zst, load_zst, save_zst, help_zst


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_JSON_ZST = os.path.join(TEST_DATA_PATH, "zst", "example_zst_{}.json.zst")
FILENAMES_PICKLE_ZST = os.path.join(TEST_DATA_PATH, "zst", "example_zst_{}.pickle.zst")

data_json = {}
data_json[0] = [1, "2", 3, -67, "tyu"]
data_json[1] = {"1": "2", "3": -67, "tyu": 3.}
data_json[2] = {'1': ["2", "33", 45], '3': -67, '35': {'3.0': "ty", "op": [[["op"], ["opa"]]]}}
data_json[3] = [["tyué#", "þ€è``\""]]

data_pickle = {}
data_pickle[0] = [1, "2", 3, -67, "tyu"]
data_pickle[1] = {"1": "2", "3": -67, "tyu": 3.}
data_pickle[2] = {'1': ["2", "33", 45], '3': -67, '35': {'3.0': "ty", "op": [[["op"], ["opa"]]]}}
data_pickle[3] = [["tyué#", "þ€è``\""]]


def test_decode_zst():
    for filenames_zst, data, decode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle]
    ):
        for i in range(len(data)):
            filename = filenames_zst.format(i)
            txt = load_bytes(filename)
            ldata = decode_zst(txt)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while decoding zst file nbr {}".format(i)


def test_encode_zst():
    for filenames_zst, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle],
            [encode_json, encode_pickle]
    ):
        for i in range(len(data)):
            txt = encode_zst(encode_func(data[i]))
            ldata = decode_zst(txt)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while encoding zst file nbr {}".format(i)


def test_read_zst():
    for filenames_zst, data, decode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle]
    ):
        for i in range(len(data)):
            filename = filenames_zst.format(i)
            with open(filename, "rb") as f:
                ldata = read_zst(f)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while reading zst file nbr {}".format(i)


def test_write_zst():
    for filenames_zst, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle],
            [encode_json, encode_pickle]
    ):
        filename = filenames_zst.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_zst(f, encode_func(data[i]))
            with open(filename, "rb") as f:
                ldata = read_zst(f)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while writing zst file nbr {}".format(i)
        os.remove(filename)


def test_load_zst():
    for filenames_zst, data, decode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle]
    ):
        for i in range(len(data)):
            filename = filenames_zst.format(i)
            ldata = load_zst(filename)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while loading zst file nbr {}".format(i)


def test_save_zst():
    for filenames_zst, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_ZST, FILENAMES_PICKLE_ZST],
            [data_json, data_pickle],
            [decode_json, decode_pickle],
            [encode_json, encode_pickle]
    ):
        filename = filenames_zst.format("write")
        for i in range(len(data)):
            save_zst(filename, encode_func(data[i]))
            ldata = load_zst(filename)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while saving zst file nbr {}".format(i)
        os.remove(filename)


def test_help_zst():
    help_zst()
