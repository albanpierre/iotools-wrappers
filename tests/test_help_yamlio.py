
import os

from iotools.txtio import load_txt
from simplified_iotools.yamlio import decode_yaml, encode_yaml, read_yaml, write_yaml, load_yaml, save_yaml


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_YAML = os.path.join(TEST_DATA_PATH, "yaml", "example_yaml_{}.yaml")

data = {}
data[0] = {'image': 'some/path/to', 'redis': {'number': 5, 'secret': '4bX-7_89.gh0'}}


def test_help_decode_yaml():
    for i in range(len(data)):
        filename = FILENAMES_YAML.format(i)
        txt = load_txt(filename)
        ldata = decode_yaml(txt)
        assert ldata == data[i], "Error while reading yaml file nbr {}".format(i)


def test_help_encode_yaml():
    for i in range(len(data)):
        txt = encode_yaml(data[i])
        ldata = decode_yaml(txt)
        assert ldata == data[i], "Error while writing yaml file nbr {}".format(i)


def test_help_read_yaml():
    for i in range(len(data)):
        filename = FILENAMES_YAML.format(i)
        with open(filename, "r") as f:
            ldata = read_yaml(f)
        assert ldata == data[i], "Error while reading yaml file nbr {}".format(i)


def test_help_write_yaml():
    filename = FILENAMES_YAML.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_yaml(f, data[i])
        with open(filename, "r") as f:
            ldata = read_yaml(f)
        assert ldata == data[i], "Error while writing yaml file nbr {}".format(i)
    os.remove(filename)


def test_help_load_yaml():
    for i in range(len(data)):
        filename = FILENAMES_YAML.format(i)
        ldata = load_yaml(filename)
        assert ldata == data[i], "Error while reading yaml file nbr {}".format(i)


def test_help_save_yaml():
    filename = FILENAMES_YAML.format("write")
    for i in range(len(data)):
        save_yaml(filename, data[i])
        ldata = load_yaml(filename)
        assert ldata == data[i], "Error while writing yaml file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_yaml():
    save_yaml("filename.yaml", data[0])
    import examples.examples_yaml  # noqa
    os.remove("filename.yaml")
