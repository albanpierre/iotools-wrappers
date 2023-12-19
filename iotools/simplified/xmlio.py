
import xmltodict  # pip install xmltodict


def decode_xml(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return xmltodict.parse(data)


def encode_xml(data):
    return xmltodict.unparse(data)


def read_xml(openfile):
    data = openfile.read()
    return xmltodict.parse(data)


def write_xml(openfile, data):
    data = xmltodict.unparse(data)
    openfile.write(data)


def load_xml(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return xmltodict.parse(data)


def save_xml(filename, data):
    data = xmltodict.unparse(data)
    with open(filename, 'w') as f:
        f.write(data)
