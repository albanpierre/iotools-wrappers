
import os

from iotools.bytesio import load_bytes
from iotools.jsonio import decode_json, encode_json
from iotools.pickleio import decode_pickle, encode_pickle
from iotools.tario import decode_tar, encode_tar
from iotools.gzio import decode_gz, encode_gz, read_gz, write_gz, load_gz, save_gz, help_gz


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_JSON_GZ = os.path.join(TEST_DATA_PATH, "gz", "example_gz_{}.json.gz")
FILENAMES_PICKLE_GZ = os.path.join(TEST_DATA_PATH, "gz", "example_gz_{}.pickle.gz")
FILENAMES_TAR_GZ = os.path.join(TEST_DATA_PATH, "gz", "example_gz_{}.tar.gz")

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

data_tar = {}
data_tar[0] = {"filename.txt": b"some binary text"}


def test_decode_gz():
    for filenames_gz, data, decode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar]
    ):
        for i in range(len(data)):
            filename = filenames_gz.format(i)
            txt = load_bytes(filename)
            ldata = decode_gz(txt)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while encoding gz file nbr {}".format(i)


def test_encode_gz():
    for filenames_gz, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar],
            [encode_json, encode_pickle, encode_tar]
    ):
        for i in range(len(data)):
            txt = encode_gz(encode_func(data[i]))
            ldata = decode_gz(txt)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while decoing gz file nbr {}".format(i)


def test_read_gz():
    for filenames_gz, data, decode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar]
    ):
        for i in range(len(data)):
            filename = filenames_gz.format(i)
            with open(filename, "rb") as f:
                ldata = read_gz(f)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while reading gz file nbr {}".format(i)


def test_write_gz():
    for filenames_gz, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar],
            [encode_json, encode_pickle, encode_tar]
    ):
        filename = filenames_gz.format("write")
        for i in range(len(data)):
            with open(filename, "wb") as f:
                write_gz(f, encode_func(data[i]))
            with open(filename, "rb") as f:
                ldata = read_gz(f)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while writing gz file nbr {}".format(i)
        os.remove(filename)


def test_load_gz():
    for filenames_gz, data, decode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar]
    ):
        for i in range(len(data)):
            filename = filenames_gz.format(i)
            ldata = load_gz(filename)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while loading gz file nbr {}".format(i)


def test_save_gz():
    for filenames_gz, data, decode_func, encode_func in zip(
            [FILENAMES_JSON_GZ, FILENAMES_PICKLE_GZ, FILENAMES_TAR_GZ],
            [data_json, data_pickle, data_tar],
            [decode_json, decode_pickle, decode_tar],
            [encode_json, encode_pickle, encode_tar]
    ):
        filename = filenames_gz.format("write")
        for i in range(len(data)):
            save_gz(filename, encode_func(data[i]))
            ldata = load_gz(filename)
            ldata = decode_func(ldata)
            assert ldata == data[i], "Error while saving gz file nbr {}".format(i)
        os.remove(filename)


def test_help_gz():
    help_gz()
