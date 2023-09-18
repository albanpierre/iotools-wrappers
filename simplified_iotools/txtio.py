
def decode_txt(data):
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def encode_txt(data):
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def read_txt(openfile):
    return openfile.read()


def write_txt(openfile, data):
    openfile.write(data)


def load_txt(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data


def save_txt(filename, data):
    with open(filename, "w") as f:
        f.write(data)
