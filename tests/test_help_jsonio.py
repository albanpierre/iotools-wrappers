
import os

from iotools.txtio import load_txt
from simplified_iotools.jsonio import decode_json, encode_json, read_json, write_json, load_json, save_json


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_JSON = os.path.join(TEST_DATA_PATH, "json", "example_json_{}.json")

data = {}
data[0] = [1, "2", 3, -67, "tyu"]
data[1] = {"1": "2", "3": -67, "tyu": 3.}
data[2] = {'1': ["2", "33", 45], '3': -67, '35': {'3.0': "ty", "op": [[["op"], ["opa"]]]}}
data[3] = [["tyué#", "þ€è``\""]]


def test_help_decode_json():
    for i in range(len(data)):
        filename = FILENAMES_JSON.format(i)
        txt = load_txt(filename)
        ldata = decode_json(txt)
        assert ldata == data[i], "Error while decoding json file nbr {}".format(i)


def test_help_encode_json():
    for i in range(len(data)):
        txt = encode_json(data[i])
        ldata = decode_json(txt)
        assert ldata == data[i], "Error while encoding json file nbr {}".format(i)


def test_help_read_json():
    for i in range(len(data)):
        filename = FILENAMES_JSON.format(i)
        with open(filename, "r") as f:
            ldata = read_json(f)
        assert ldata == data[i], "Error while reading json file nbr {}".format(i)


def test_help_write_json():
    filename = FILENAMES_JSON.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_json(f, data[i])
        with open(filename, "r") as f:
            ldata = read_json(f)
        assert ldata == data[i], "Error while writing json file nbr {}".format(i)
    os.remove(filename)


def test_help_load_json():
    for i in range(len(data)):
        filename = FILENAMES_JSON.format(i)
        ldata = load_json(filename)
        assert ldata == data[i], "Error while loading json file nbr {}".format(i)


def test_help_save_json():
    filename = FILENAMES_JSON.format("write")
    for i in range(len(data)):
        save_json(filename, data[i])
        ldata = load_json(filename)
        assert ldata == data[i], "Error while saving json file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_json():
    save_json("filename.json", data[0])
    import examples.examples_json  # noqa
    os.remove("filename.json")
