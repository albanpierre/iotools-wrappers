
import os

from iotools.bytesio import load_bytes
from iotools.simplified.pickleio import (
    decode_pickle, encode_pickle, read_pickle, write_pickle, load_pickle, save_pickle
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_PICKLE = os.path.join(TEST_DATA_PATH, "pickle", "example_pickle_{}.pickle")

data = {}
data[0] = [1, "2", 3, -67, "tyu"]
data[1] = {"1": "2", "3": -67, "tyu": 3.}
data[2] = {'1': ["2", "33", 45], '3': -67, '35': {'3.0': "ty", "op": [[["op"], ["opa"]]]}}
data[3] = [["tyué#", "þ€è``\""]]


def test_help_decode_pickle():
    for i in range(len(data)):
        filename = FILENAMES_PICKLE.format(i)
        txt = load_bytes(filename)
        ldata = decode_pickle(txt)
        assert ldata == data[i], "Error while reading pickle file nbr {}".format(i)


def test_help_encode_pickle():
    for i in range(len(data)):
        txt = encode_pickle(data[i])
        ldata = decode_pickle(txt)
        assert ldata == data[i], "Error while writing pickle file nbr {}".format(i)


def test_help_read_pickle():
    for i in range(len(data)):
        filename = FILENAMES_PICKLE.format(i)
        with open(filename, "rb") as f:
            ldata = read_pickle(f)
        assert ldata == data[i], "Error while reading pickle file nbr {}".format(i)


def test_help_write_pickle():
    filename = FILENAMES_PICKLE.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_pickle(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_pickle(f)
        assert ldata == data[i], "Error while writing pickle file nbr {}".format(i)
    os.remove(filename)


def test_help_load_pickle():
    for i in range(len(data)):
        filename = FILENAMES_PICKLE.format(i)
        ldata = load_pickle(filename)
        assert ldata == data[i], "Error while reading pickle file nbr {}".format(i)


def test_help_save_pickle():
    filename = FILENAMES_PICKLE.format("write")
    for i in range(len(data)):
        save_pickle(filename, data[i])
        ldata = load_pickle(filename)
        assert ldata == data[i], "Error while writing pickle file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_pickle():
    save_pickle("filename.pickle", data[0])
    import iotools.examples.examples_pickle  # noqa
    os.remove("filename.pickle")
