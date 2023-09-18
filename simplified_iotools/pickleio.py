
import pickle


def decode_pickle(data):
    return pickle.loads(data)


def encode_pickle(data):
    return pickle.dumps(data)


def read_pickle(openfile):
    return pickle.load(openfile)


def write_pickle(openfile, data):
    return pickle.dump(data, openfile)


def load_pickle(filename, fs=None):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
