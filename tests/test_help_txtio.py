
import os

from iotools.simplified.txtio import decode_txt, encode_txt, read_txt, write_txt, load_txt, save_txt


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_TXT = os.path.join(TEST_DATA_PATH, "txt", "example_txt_{}.txt")

data = {}
data[0] = "1, 2, 3, -67, tyu cvd"
data[1] = "1, \n 2, 3\t, -67\n\n, tyu cvd\n\n"
data[2] = "t[[yué#\", \\\"''þ€è``\"}}"


def test_help_decode_txt():
    for i in range(len(data)):
        ldata = decode_txt(data[i])
        assert ldata == data[i], "Error while decoding txt file nbr {}".format(i)
        ldata = decode_txt(data[i].encode("utf-8"))
        assert ldata == data[i], "Error while decoding txt file nbr {}".format(i)


def test_help_encode_txt():
    for i in range(len(data)):
        ldata = encode_txt(data[i])
        assert ldata == data[i], "Error while decoding txt file nbr {}".format(i)
        ldata = encode_txt(data[i].encode("utf-8"))
        assert ldata == data[i], "Error while decoding txt file nbr {}".format(i)


def test_help_read_txt():
    for i in range(len(data)):
        filename = FILENAMES_TXT.format(i)
        with open(filename, "r", encoding="utf-8") as f:
            ldata = read_txt(f)
        assert ldata == data[i], "Error while reading txt file nbr {}".format(i)


def test_help_write_txt():
    filename = FILENAMES_TXT.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_txt(f, data[i])
        with open(filename, "r") as f:
            ldata = read_txt(f)
        assert ldata == data[i], "Error while writing txt file nbr {}".format(i)
    os.remove(filename)


def test_help_load_txt():
    for i in range(len(data) - 1):  # avoid encoding errors in windows
        filename = FILENAMES_TXT.format(i)
        ldata = load_txt(filename)
        assert ldata == data[i], "Error while loading txt file nbr {}".format(i)


def test_help_save_txt():
    filename = FILENAMES_TXT.format("save")
    for i in range(len(data)):
        save_txt(filename, data[i])
        ldata = load_txt(filename)
        assert ldata == data[i], "Error while saving txt file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_txt():
    save_txt("filename.txt", data[0])
    import iotools.examples.examples_txt  # noqa
    os.remove("filename.txt")
