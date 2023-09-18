
import csv
from io import StringIO


def decode_csv_using_csv(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    filelike = StringIO(data)
    reader = csv.reader(filelike)
    ret = list(reader)
    filelike.close()
    return ret


def encode_csv_using_csv(data):
    filelike = StringIO()
    writer = csv.writer(filelike)
    for row in data:
        writer.writerow(row)
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_csv_using_csv(openfile):
    reader = csv.reader(openfile)
    return list(reader)


def write_csv_using_csv(openfile, data):
    writer = csv.writer(openfile)
    for row in data:
        writer.writerow(row)


def load_csv_using_csv(filename):
    with open(filename, 'r') as openfile:
        reader = csv.reader(openfile)
        data = list(reader)
    return data


def save_csv_using_csv(filename, data):
    with open(filename, 'w') as openfile:
        writer = csv.writer(openfile)
        for row in data:
            writer.writerow(row)
