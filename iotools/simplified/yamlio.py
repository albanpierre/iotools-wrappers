
import yaml  # pip install PyYAML


def decode_yaml(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return yaml.safe_load(data)


def encode_yaml(data):
    return yaml.safe_dump(data)


def read_yaml(openfile):
    return yaml.safe_load(openfile)


def write_yaml(openfile, data):
    return yaml.safe_dump(data, openfile)


def load_yaml(filename, fs=None):
    with open(filename, 'r') as f:
        data = read_yaml(f)
    return data


def save_yaml(filename, data):
    with open(filename, 'w') as f:
        write_yaml(f, data)
