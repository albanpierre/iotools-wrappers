
import json


def decode_json(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def encode_json(data):
    return json.dumps(data)


def read_json(openfile):
    return json.load(openfile)


def write_json(openfile, data):
    json.dump(data, openfile)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
