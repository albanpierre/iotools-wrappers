
import pandas as pd  # pip install pandas
from io import StringIO


def decode_csv_using_pandas(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    filelike = StringIO(data)
    ret = pd.read_csv(filelike)
    filelike.close()
    return ret


def encode_csv_using_pandas(data):
    filelike = StringIO()
    pd.DataFrame(data).to_csv(filelike)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_csv_using_pandas(openfile):
    return pd.read_csv(openfile)


def write_csv_using_pandas(openfile, data):
    pd.DataFrame(data).to_csv(openfile)


def load_csv_using_pandas(filename):
    return pd.read_csv(filename)


def save_csv_using_pandas(filename, data):
    pd.DataFrame(data).to_csv(filename)
