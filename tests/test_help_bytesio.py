
import os

from iotools.simplified.bytesio import decode_bytes, encode_bytes, read_bytes, write_bytes, load_bytes, save_bytes


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_BYTES = os.path.join(TEST_DATA_PATH, "bytes", "example_bytes_{}.bytes")

data = {}
data[0] = "1, 2, 3, -67, tyu cvd".encode('utf-8')
data[1] = "1, \n 2, 3\t, -67\n\n, tyu cvd\n\n".encode('utf-8')
data[2] = "t[[yué#\", \\\"''þ€è``\"}}".encode('utf-8')


def test_help_decode_bytes():
    for i in range(len(data)):
        ldata = decode_bytes(data[i])
        assert ldata == data[i], "Error while decoding bytes file nbr {}".format(i)
        ldata = decode_bytes(data[i].decode("utf-8"))
        assert ldata == data[i], "Error while decoding bytes file nbr {}".format(i)


def test_help_encode_bytes():
    for i in range(len(data)):
        ldata = encode_bytes(data[i])
        assert ldata == data[i], "Error while decoding bytes file nbr {}".format(i)
        ldata = encode_bytes(data[i].decode("utf-8"))
        assert ldata == data[i], "Error while decoding bytes file nbr {}".format(i)


def test_help_read_bytes():
    for i in range(len(data)):
        filename = FILENAMES_BYTES.format(i)
        with open(filename, "rb") as f:
            ldata = read_bytes(f)
        assert ldata == data[i], "Error while reading bytes file nbr {}".format(i)


def test_help_write_bytes():
    filename = FILENAMES_BYTES.format("write")
    for i in range(len(data)):
        with open(filename, "wb") as f:
            write_bytes(f, data[i])
        with open(filename, "rb") as f:
            ldata = read_bytes(f)
        assert ldata == data[i], "Error while writing bytes file nbr {}".format(i)
    os.remove(filename)


def test_help_load_bytes():
    for i in range(len(data)):
        filename = FILENAMES_BYTES.format(i)
        ldata = load_bytes(filename)
        assert ldata == data[i], "Error while loading bytes file nbr {}".format(i)


def test_help_save_bytes():
    filename = FILENAMES_BYTES.format("save")
    for i in range(len(data)):
        save_bytes(filename, data[i])
        ldata = load_bytes(filename)
        assert ldata == data[i], "Error while saving bytes file nbr {}".format(i)
    os.remove(filename)


def test_help_examples_bytes():
    save_bytes("filename.bytes", data[0])
    import iotools.examples.examples_bytes  # noqa
    os.remove("filename.bytes")
