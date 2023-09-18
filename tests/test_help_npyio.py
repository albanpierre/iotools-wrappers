
import os
import numpy as np

from iotools.bytesio import load_bytes
from simplified_iotools.npyio import decode_npy, encode_npy, read_npy, write_npy, load_npy, save_npy


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_NPY = os.path.join(TEST_DATA_PATH, "npy", "example_npy_{}.npy")

data = {}
data[0] = np.asarray([1, 3])
data[1] = np.asarray([[1, 2, 3.6], [4, 7.6, 8.89]])
data[2] = np.asarray([[1, 2, None], [None, "ty", 8.89]])


def test_help_decode_npy():
    for i in range(len(data)):
        filename = FILENAMES_NPY.format(i)
        txt = load_bytes(filename)
        ldata = decode_npy(txt, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while encoding npy file nbr {}".format(i)


def test_help_encode_npy():
    for i in range(len(data)):
        txt = encode_npy(data[i])
        ldata = decode_npy(txt, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while decoding npy file nbr {}".format(i)


def test_help_read_npy():
    for i in range(len(data)):
        filename = FILENAMES_NPY.format(i)
        with open(filename, "rb") as f:
            ldata = read_npy(f, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while reading npy file nbr {}".format(i)


def test_help_write_npy():
    filename = FILENAMES_NPY.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_npy(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_npy(f, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while writing npy file nbr {}".format(i)
    os.remove(filename)


def test_help_load_npy():
    for i in range(len(data)):
        filename = FILENAMES_NPY.format(i)
        ldata = load_npy(filename, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while loading npy file nbr {}".format(i)


def test_help_save_npy():
    filename = FILENAMES_NPY.format("write")
    for i in range(len(data)):
        save_npy(filename, data[i])
        ldata = load_npy(filename, allow_pickle=True)
        assert np.all(ldata == data[i]), "Error while saving npy file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_npy():
    save_npy("filename.npy", data[0])
    import examples.examples_npy  # noqa
    os.remove("filename.npy")
