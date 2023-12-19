
import os


def decode_bytes(data):
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def encode_bytes(data):
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def read_bytes(openfile):
    data = openfile.read()
    if (os.name == "nt"):  # On windows
        data = data.replace(b'\r\n', b'\n')  # Windows has different newlines encoding
    return data


def write_bytes(openfile, data):
    openfile.write(data)


def load_bytes(filename):
    with open(filename, "rb") as f:
        data = f.read()
    return data


def save_bytes(filename, data):
    with open(filename, "wb") as f:
        f.write(data)
