
import gzip


def decode_gz(data):
    return gzip.decompress(data)


def encode_gz(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return gzip.compress(data)


def read_gz(openfile):
    return gzip.decompress(openfile.read())


def write_gz(openfile, data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    openfile.write(gzip.compress(data))


def load_gz(filename):
    with open(filename, 'rb') as openfile:
        data = gzip.decompress(openfile.read())
    return data


def save_gz(filename, data):
    with open(filename, 'wb') as openfile:
        if isinstance(data, str):
            data = data.encode("utf-8")
        openfile.write(gzip.compress(data))
