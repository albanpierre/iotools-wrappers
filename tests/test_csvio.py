
import os

from iotools.txtio import load_txt
from iotools.csvio import (
    decode_csv_using_csv, encode_csv_using_csv,
    read_csv_using_csv, write_csv_using_csv,
    load_csv_using_csv, save_csv_using_csv, help_csv_using_csv,
    decode_csv_using_pandas, encode_csv_using_pandas,
    read_csv_using_pandas, write_csv_using_pandas,
    load_csv_using_pandas, save_csv_using_pandas, help_csv_using_pandas,
    decode_csv, encode_csv, read_csv, write_csv, load_csv, save_csv, help_csv
)


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_CSV = os.path.join(TEST_DATA_PATH, "csv", "example_csv_{}.csv")

data = {}
data[0] = [[1, "a", 3, -67, "tyu"]]
data[1] = [[0, 2], ["rt", "bhj"], [4.71, -7], [3.82, "fh"]]
data[2] = [[1, ""], ["bh", "ty"], [-3, ""]]
data[3] = [["tyué#", "þ€è``\""]]


# +-----------+
# | Using csv |
# +-----------+


def test_decode_csv_using_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        txt = load_txt(filename)
        ldata = decode_csv_using_csv(txt)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while decoding csv file nbr {}".format(i)


def test_encode_csv_using_csv():
    for i in range(len(data)):
        txt = encode_csv_using_csv(data[i])
        ldata = decode_csv_using_csv(txt)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while encoding csv file nbr {}".format(i)


def test_read_csv_using_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        with open(filename, "r") as f:
            ldata = read_csv_using_csv(f)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while reading csv file nbr {}".format(i)


def test_write_csv_using_csv():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_csv_using_csv(f, data[i])
        with open(filename, "r") as f:
            ldata = read_csv_using_csv(f)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while writing csv file nbr {}".format(i)
    os.remove(filename)


def test_load_csv_using_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        ldata = load_csv_using_csv(filename)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while loading csv file nbr {}".format(i)


def test_save_csv_using_csv():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        save_csv_using_csv(filename, data[i])
        ldata = load_csv_using_csv(filename)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while saving csv file nbr {}".format(i)
    os.remove(filename)


def test_help_csv_using_csv():
    help_csv_using_csv()


# +--------------+
# | Using pandas |
# +--------------+


def test_decode_csv_using_pandas():
    for i in range(len(data)):
        filename = FILENAMES_CSV.format(i)
        txt = load_txt(filename)
        _ = decode_csv_using_pandas(txt)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)


def test_encode_csv_using_pandas():
    for i in range(len(data)):
        txt = encode_csv_using_pandas(data[i])
        _ = decode_csv_using_pandas(txt)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)


def test_read_csv_using_pandas():
    for i in range(1, len(data)):
        filename = FILENAMES_CSV.format(i)
        with open(filename, "r") as f:
            _ = read_csv_using_pandas(f)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)


def test_write_csv_using_pandas():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_csv_using_pandas(f, data[i])
        with open(filename, "r") as f:
            _ = read_csv_using_pandas(f)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)
    os.remove(filename)


def test_load_csv_using_pandas():
    for i in range(len(data)):
        filename = FILENAMES_CSV.format(i)
        _ = load_csv_using_pandas(filename)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)


def test_save_csv_using_pandas():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        save_csv_using_pandas(filename, data[i])
        _ = load_csv_using_pandas(filename)
        # Pandas infers too many things on the csv (modifying data too much to test it afterwards)
    os.remove(filename)


def test_help_csv_using_pandas():
    help_csv_using_pandas()


# +------------------+
# | Default behavior |
# +------------------+


def test_decode_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        txt = load_txt(filename)
        ldata = decode_csv(txt)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while decoding csv file nbr {}".format(i)


def test_encode_csv():
    for i in range(len(data)):
        txt = encode_csv(data[i])
        ldata = decode_csv(txt)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while encoding csv file nbr {}".format(i)


def test_read_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        with open(filename, "r") as f:
            ldata = read_csv(f)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while reading csv file nbr {}".format(i)


def test_write_csv():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_csv(f, data[i])
        with open(filename, "r") as f:
            ldata = read_csv(f)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while writing csv file nbr {}".format(i)
    os.remove(filename)


def test_load_csv():
    for i in range(len(data) - 1):  # avoid encoding errors for windows
        filename = FILENAMES_CSV.format(i)
        ldata = load_csv(filename)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while loading csv file nbr {}".format(i)


def test_save_csv():
    filename = FILENAMES_CSV.format("write")
    for i in range(len(data)):
        save_csv(filename, data[i])
        ldata = load_csv(filename)
        assert ldata == [[str(xi) for xi in x] for x in data[i]], "Error while saving csv file nbr {}".format(i)
    os.remove(filename)


def test_help_csv():
    help_csv()
