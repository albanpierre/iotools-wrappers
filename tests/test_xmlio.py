
import os
from collections import OrderedDict

from iotools.txtio import load_txt
from iotools.xmlio import decode_xml, encode_xml, read_xml, write_xml, load_xml, save_xml, help_xml


TEST_DATA_PATH = "tests/data_for_tests/"

FILENAMES_XML = os.path.join(TEST_DATA_PATH, "xml", "example_xml_{}.xml")

data = {}
data[0] = OrderedDict([
    (
        'note',
        OrderedDict([
            ('to', 'Tove'),
            ('from', 'Jani'),
            ('heading', 'Reminder'),
            ('body', "Don't forget me this weekend!")
        ])
    )
])


def test_decode_xml():
    for i in range(len(data)):
        filename = FILENAMES_XML.format(i)
        txt = load_txt(filename)
        ldata = decode_xml(txt)
        assert ldata == data[i], "Error while reading xml file nbr {}".format(i)


def test_encode_xml():
    for i in range(len(data)):
        txt = encode_xml(data[i])
        ldata = decode_xml(txt)
        assert ldata == data[i], "Error while writing xml file nbr {}".format(i)


def test_read_xml():
    for i in range(len(data)):
        filename = FILENAMES_XML.format(i)
        with open(filename, "r") as f:
            ldata = read_xml(f)
        assert ldata == data[i], "Error while reading xml file nbr {}".format(i)


def test_write_xml():
    filename = FILENAMES_XML.format("write")
    for i in range(len(data)):
        with open(filename, "w") as f:
            write_xml(f, data[i])
        with open(filename, "r") as f:
            ldata = read_xml(f)
        assert ldata == data[i], "Error while writing xml file nbr {}".format(i)
    os.remove(filename)


def test_load_xml():
    for i in range(len(data)):
        filename = FILENAMES_XML.format(i)
        ldata = load_xml(filename)
        assert ldata == data[i], "Error while reading xml file nbr {}".format(i)


def test_save_xml():
    filename = FILENAMES_XML.format("write")
    for i in range(len(data)):
        save_xml(filename, data[i])
        ldata = load_xml(filename)
        assert ldata == data[i], "Error while writing xml file nbr {}".format(i)
    os.remove(filename)


def test_help_xml():
    help_xml()
